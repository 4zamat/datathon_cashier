import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from annotation.app import create_app
from annotation.models import OBJECT_CLASSES, YOLO_CLASSES, EQUIPMENT_CLASSES, Frame
from tests.helpers import make_video


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.data = Path(self.temporary.name)
        (self.data / "videos").mkdir()
        self.source = self.data / "videos/Х-1.mov"
        self.original = make_video(self.source)
        (self.data / "labels.csv").write_text("video,has_violation,timecode_sec\nvideos/Х-1.mov,1,0.5\n", encoding="utf-8")
        self.original_labels = (self.data / "labels.csv").read_bytes()
        self.app = create_app(self.data)
        self.client = TestClient(self.app)
        self.headers = {"X-Annotation-Token": self.client.get("/api/config").json()["token"]}
        self.video = self.client.get("/api/videos").json()[0]
        self.base = f"/api/videos/{self.video['id']}"

    def tearDown(self):
        self.client.close()
        self.temporary.cleanup()

    def document(self):
        document = self.client.get(self.base + "/annotation").json()
        document["reviewer"] = "Тест"
        return document

    def capture(self):
        response = self.client.post(self.base + "/capture", headers=self.headers, json={"timestamp_sec": 0.5})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def save(self, document):
        return self.client.put(self.base + "/annotation", headers=self.headers, json=document)

    def test_roundtrip_conflict_history_and_source_preservation(self):
        document = self.document()
        document["events"] = [{"id": "event1", "start_sec": 0.2, "end_sec": 0.7,
                               "event_type": "unknown", "assessment": "uncertain", "observation": "Рука закрывает предмет"}]
        document["event_draft"] = {"start": "0.4", "end": "", "observation": "Ещё проверяю"}
        first = self.save(document)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json()["revision"], 1)
        self.assertEqual(self.save(document).status_code, 409)
        second = first.json()
        second["notes"] = "Уточнение"
        self.assertEqual(self.save(second).json()["revision"], 2)
        self.assertEqual(len(list((self.data / "annotations/local/history").rglob("*.json"))), 1)
        # Reconstruct the app: persistence cannot depend on its in-memory state.
        with TestClient(create_app(self.data)) as restarted:
            loaded = restarted.get(self.base + "/annotation").json()
            self.assertEqual(loaded["events"], document["events"])
            self.assertEqual(loaded["event_draft"]["observation"], "Ещё проверяю")
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual((self.data / "labels.csv").read_bytes(), self.original_labels)

    def test_invalid_intervals_types_and_frame_geometry_are_rejected(self):
        document = self.document()
        event = {"id": "a", "start_sec": 0.6, "end_sec": 0.3, "observation": "Видимое действие"}
        document["events"] = [event]
        self.assertEqual(self.save(document).status_code, 422)
        event.update(start_sec=0.1, end_sec=self.video["duration_sec"] + 1)
        self.assertEqual(self.save(document).status_code, 422)
        event.update(end_sec=0.5, event_type="demo_violation")
        self.assertEqual(self.save(document).status_code, 422)
        document["events"] = []
        frame = self.capture()
        document["frames"] = [frame]
        frame["boxes"] = [{"id": "b", "label": "cash", "x": 0.9, "y": 0.1, "width": 0.3, "height": 0.1}]
        self.assertEqual(self.save(document).status_code, 422)
        frame["boxes"] = []
        frame["width"] += 1
        self.assertEqual(self.save(document).status_code, 422)
        frame["width"] -= 1
        document["frames"] = [frame, frame]
        self.assertEqual(self.save(document).status_code, 422)
        document["frames"] = [frame]
        frame["boxes"] = [{"id": "b", "label": "other", "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.1}]
        frame["reviewed"] = True
        self.assertEqual(self.save(document).status_code, 422)

    def test_local_origin_token_and_paths(self):
        self.assertEqual(self.client.put(self.base + "/annotation", json=self.document()).status_code, 403)
        self.assertEqual(self.client.put(self.base + "/annotation", json=self.document(),
                                        headers={**self.headers, "Origin": "https://elsewhere.example"}).status_code, 403)
        self.assertEqual(self.client.get("/api/videos", headers={"Host": "elsewhere.example"}).status_code, 400)
        self.assertEqual(self.client.get("/media/not-a-video").status_code, 404)
        self.assertEqual(self.client.get(f"/frames/{self.video['id']}/f999.jpg").status_code, 404)
        self.assertEqual(self.client.get("/assets/../workspace.py").status_code, 404)

    def test_export_frames_yolo_geometry_and_draft_separation(self):
        document = self.document()
        frame = self.capture()
        frame["boxes"] = [{"id": "box", "label": "cash", "x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4,
                           "occluded": True, "comment": "Часть скрыта"}]
        document["frames"] = [frame]
        document["events"] = [{"id": "e", "start_sec": 0, "end_sec": 0.5,
                               "event_type": "unknown", "assessment": "uncertain", "observation": "Предмет не различим"}]
        saved = self.save(document)
        self.assertEqual(saved.status_code, 200, saved.text)
        response = self.client.post("/api/export", headers=self.headers)
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            self.assertEqual(len(list(csv.DictReader(io.StringIO(archive.read("events.csv").decode("utf-8-sig"))))), 0)
            self.assertEqual(len(list(csv.DictReader(io.StringIO(archive.read("events_draft.csv").decode("utf-8-sig"))))), 1)
            self.assertFalse(any(name.startswith("yolo/images/") for name in archive.namelist()))
        document = saved.json()
        document["scene_group_id"] = "scene_01"
        document["frames"][0]["reviewed"] = True
        document["frames"][0]["reviewed_classes"] = YOLO_CLASSES
        self.assertEqual(self.save(document).status_code, 200)
        response = self.client.post("/api/export", headers=self.headers)
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            label_name = next(name for name in archive.namelist() if name.startswith("yolo/labels/"))
            fields = archive.read(label_name).decode().split()
            self.assertEqual(int(fields[0]), OBJECT_CLASSES.index("cash"))
            self.assertEqual(list(map(float, fields[1:])), [0.25, 0.4, 0.3, 0.4])
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest["yolo_reviewed_frames"], 1)
            self.assertFalse(manifest["training_ready"])
            event = list(csv.DictReader(io.StringIO(archive.read("events.csv").decode("utf-8-sig"))))[0]
            self.assertEqual(event["scene_group_id"], "scene_01")
            self.assertEqual(event["video"], "videos/Х-1.mov")
            self.assertFalse(any(name.endswith(".mov") for name in archive.namelist()))

    def test_equipment_review_is_not_full_object_review(self):
        document = self.document()
        frame = self.capture()
        frame.update(reviewed=True, reviewed_classes=EQUIPMENT_CLASSES, reviewed_by="assistant_visual_review")
        frame["boxes"] = [{"id": "laptop1", "label": "laptop", "x": 0.1, "y": 0.2,
                           "width": 0.3, "height": 0.2, "source": "assistant", "annotator": "assistant_visual_review"}]
        document["frames"] = [frame]
        self.assertEqual(self.save(document).status_code, 200)
        response = self.client.post("/api/export", headers=self.headers)
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            self.assertFalse(any(name.startswith("yolo/labels/") for name in archive.namelist()))
            labels = [name for name in archive.namelist() if name.startswith("equipment_yolo/labels/")]
            self.assertEqual(len(labels), 1)
            self.assertEqual(int(archive.read(labels[0]).split()[0]), EQUIPMENT_CLASSES.index("laptop"))
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest["yolo_reviewed_frames"], 0)
            self.assertEqual(manifest["equipment_reviewed_frames"], 1)
        legacy = Frame(id="f0", frame_index=0, timestamp_sec=0, width=64, height=48, reviewed=True)
        self.assertNotIn("laptop", legacy.reviewed_classes)

    def test_missing_reviewer_and_changed_video_do_not_overwrite(self):
        document = self.document()
        document.update(reviewer="", watched_full=True)
        self.assertEqual(self.save(document).status_code, 422)
        document.update(reviewer="Тест", watched_full=False)
        self.assertEqual(self.save(document).status_code, 200)
        # Simulate source replacement while retaining a decodable file.
        with self.source.open("ab") as stream:
            stream.write(b"different source")
        self.assertEqual(self.client.get(self.base + "/annotation").status_code, 409)
        with TestClient(create_app(self.data)) as restarted:
            self.assertEqual(restarted.get(self.base + "/annotation").status_code, 409)


if __name__ == "__main__":
    unittest.main()
