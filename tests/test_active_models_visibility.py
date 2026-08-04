# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from omlx.admin import routes as admin_routes


class FakePool:
    def __init__(self, scheduler, loading_started_at=None):
        core = SimpleNamespace(
            _output_collectors={"gen-1": object(), "prefill-1": object(), "wait-1": object()},
            scheduler=scheduler,
        )
        engine = SimpleNamespace(_engine=SimpleNamespace(engine=core))
        self._entries = {"model-a": SimpleNamespace(engine=engine)}
        self.loading_started_at = loading_started_at

    def get_status(self):
        return {
            "current_model_memory": 1024,
            "final_ceiling": 2048,
            "models": [
                {
                    "id": "model-a",
                    "loaded": True,
                    "is_loading": self.loading_started_at is not None,
                    "loading_started_at": self.loading_started_at,
                    "estimated_size": 1024,
                    "pinned": False,
                }
            ],
        }


class FakePrefillTracker:
    def get_model_progress(self, model_id):
        assert model_id == "model-a"
        return [{"request_id": "prefill-1", "processed": 10, "total": 20}]


def test_active_models_generation_includes_activity_and_waiting_rows():
    running_request = SimpleNamespace(
        request_id="gen-1",
