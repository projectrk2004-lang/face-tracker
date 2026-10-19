
from src.utils import crop_face


class VisitorTracker:
    """Decides ENTRY / EXIT events and registers or recognizes faces."""

    def __init__(self, cfg, db, recognizer, logger, save_image):
        self.cfg = cfg
        self.db = db
        self.rec = recognizer
        self.log = logger
        self.save_image = save_image

        self.known = db.get_all_faces()
        self.state = {}
        self._announced = set()

    # --------------------------------------------------------------
    def update(self, frame, detections, frame_no):
        """Call once per detection cycle with the current detections."""

        for tid, box, conf in detections:
            crop = crop_face(frame, box)

            if crop is None:
                continue

            if tid in self.state:
                s = self.state[tid]
                s["last_seen"] = frame_no
                s["last_crop"] = crop
                s["updates"] += 1

                if s["updates"] % self.cfg["tracking_log_every"] == 0:
                    self.log.info(
                        f"TRACKING | face_id={s['face_id']} | "
                        f"track={tid} | frame={frame_no}"
                    )
            else:
                self._handle_new_track(tid, crop, frame_no)

        self._handle_exits(frame_no)

    # --------------------------------------------------------------
    def _handle_new_track(self, tid, crop, frame_no):
        if tid not in self._announced:
            self._announced.add(tid)
            self.log.info(
                f"FACE_DETECTED | track={tid} | frame={frame_no}"
            )

        emb = self.rec.get_embedding(crop)

        if emb is None:
            self.log.info(
                f"EMBEDDING_FAILED | track={tid} | will retry"
            )
            return

        self.log.info(
            f"EMBEDDING_GENERATED | track={tid} | size={len(emb)}"
        )

        face_id, score = self.rec.match(emb, self.known)

        if face_id is None:
            face_id = self.db.add_face(emb)
            self.known[face_id] = emb

            self.log.info(
                f"FACE_REGISTERED | face_id={face_id} | "
                f"unique_count={self.db.get_unique_count()}"
            )
        else:
            self.log.info(
                f"FACE_RECOGNIZED | face_id={face_id} | "
                f"score={score:.2f}"
            )

        # Check whether this person already has an active track.
        old_tid = next(
            (
                t for t, s in self.state.items()
                if s["face_id"] == face_id
            ),
            None
        )

        if old_tid is not None:
            s = self.state.pop(old_tid)
            s["last_seen"] = frame_no
            s["last_crop"] = crop
            self.state[tid] = s

            self.log.info(
                f"TRACK_RELINKED | face_id={face_id} | "
                f"old_track={old_tid} | new_track={tid}"
            )
            return

        path = self.save_image(crop, "entry", face_id)
        self.db.add_event(face_id, "entry", path)

        self.log.info(
            f"ENTRY | face_id={face_id} | track={tid} | "
            f"image={path}"
        )

        self.state[tid] = {
            "face_id": face_id,
            "last_seen": frame_no,
            "last_crop": crop,
            "updates": 0
        }

    # --------------------------------------------------------------
    def _handle_exits(self, frame_no, force=False):
        limit = self.cfg["exit_timeout_frames"]

        for tid in list(self.state):
            s = self.state[tid]

            if force or frame_no - s["last_seen"] > limit:
                path = self.save_image(
                    s["last_crop"], "exit", s["face_id"]
                )

                self.db.add_event(s["face_id"], "exit", path)

                self.log.info(
                    f"EXIT | face_id={s['face_id']} | "
                    f"track={tid} | image={path}"
                )

                del self.state[tid]

    # --------------------------------------------------------------
    def flush(self, frame_no):
        """Log EXIT for everyone still active at video end or shutdown."""
        self._handle_exits(frame_no, force=True)
