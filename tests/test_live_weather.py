import os
import unittest

from mini_agent.models import Session
from mini_agent.tools import build_default_tools


class LiveWeatherTest(unittest.TestCase):
    def test_shanghai_weather_uses_live_source_when_enabled(self) -> None:
        if os.getenv("RUN_LIVE_TESTS") != "1":
            self.skipTest("set RUN_LIVE_TESTS=1 to run live weather test")
        session = Session(session_id="s", user_id="u", window_id="w")
        result = build_default_tools().execute("weather", {"city": "上海", "date": "today"}, session)
        self.assertEqual(result["source"], "live_open_meteo")
        self.assertIn("temperature_min_c", result)
        self.assertIn("temperature_max_c", result)


if __name__ == "__main__":
    unittest.main()
