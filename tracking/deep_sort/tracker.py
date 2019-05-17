# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
from utilities import img_transform, matching
from re_id.reid.feature_extraction.cnn import extract_cnn_feature


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=20000, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections, reid_model=None, orig_img=None, threshold=1.0):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.
        reid_model: re-identification model for counting person
        orig_img: original frame for extracting image of detection box
        threshold: matching threshold for person re-identification

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        if reid_model and len(self.tracks) and len(unmatched_detections):
            self.filter_by_re_id(detections, unmatched_detections, unmatched_tracks, orig_img, reid_model, threshold)

        # Update track set.
        for track_idx, detection_idx in matches:
            prev_img = None
            if orig_img is not None:
                x1, y1, x2, y2 = detections[detection_idx].to_tlbr()
                prev_img = orig_img[y1:y2, x1:x2, :]

            self.tracks[track_idx].update(
                self.kf, detections[detection_idx], prev_img)

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx], orig_img)
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection, img):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        prev_img = None
        if img is not None:
            x1, y1, x2, y2 = detection.to_tlbr()
            prev_img = img[y1:y2, x1:x2, :]

        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature, prev_img))
        self._next_id += 1

    def filter_by_re_id(self, detections, unmatched_detections, unmatched_tracks, orig_img, reid_model, threshold):
        query = []
        for unmatched_det_idx in unmatched_detections:
            x1, y1, x2, y2 = detections[unmatched_det_idx].to_tlbr()
            query.append(orig_img[y1:y2, x1:x2, :])

        query_imgs = img_transform(query, (128, 384))

        # creates gallery images
        gallery = []
        for unmatched_track_idx in range(len(self.tracks)):
            track = self.tracks[unmatched_track_idx]
            gallery.append(track.prev_img)

        gallery_imgs = img_transform(gallery, (128, 384))

        query_embedding = list(extract_cnn_feature(reid_model, query_imgs))
        embeddings = list(extract_cnn_feature(reid_model, gallery_imgs))

        bb_idx = matching(query_embedding, embeddings, threshold)

        for query_idx, gallery_idx in bb_idx:
            x1, y1, x2, y2 = detections[unmatched_detections[query_idx]].to_tlbr()
            self.tracks[gallery_idx].update(
                self.kf, detections[unmatched_detections[query_idx]], orig_img[y1:y2, x1:x2, :])

        tmp_bb_idx = [[unmatched_detections[query_idx], gallery_idx] for query_idx, gallery_idx in bb_idx]
        for query_val, gallery_idx in tmp_bb_idx:
            if gallery_idx in unmatched_tracks:
                unmatched_tracks.remove(gallery_idx)
            unmatched_detections.remove(query_val)
