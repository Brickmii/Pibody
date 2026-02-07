"""
PBAI Constraint Diagnostic Client
Run on PC to test Pi constraint system

Uses WebSockets to match body_server protocol.

Usage:
    pip install websockets
    python manifold_test.py
"""

import asyncio
import json
import time

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    print("Install websockets: pip install websockets")

PI_HOST = "192.168.5.22"  # or IP
PI_PORT = 8421


class DiagnosticClient:
    def __init__(self, host=PI_HOST, port=PI_PORT):
        self.host = host
        self.port = port
        self.ws = None
        self.t_K_offset = 0  # PC time offset from Pi t_K

    async def connect(self):
        uri = f"ws://{self.host}:{self.port}"
        self.ws = await websockets.connect(uri)
        print(f"✓ Connected to {uri}")

    async def disconnect(self):
        if self.ws:
            await self.ws.close()

    async def send(self, msg: dict) -> dict:
        await self.ws.send(json.dumps(msg))
        response = await self.ws.recv()
        return json.loads(response)

    async def sync_clock(self) -> bool:
        """Sync PC time to Pi t_K"""
        resp = await self.send({"type": "status"})
        pi_t_K = resp.get("t_K", 0)
        self.t_K_offset = pi_t_K - time.time()
        print(f"✓ Clock synced: Pi t_K={pi_t_K}")
        return True

    async def test_spec_creation(self) -> bool:
        """Test: Create a spec from observation (Robinson minimum)"""
        resp = await self.send({
            "type": "test_constraint",
            "action": "create_robinson_spec",
            "name": "test_duration",
            "observed": 1.6,
            "unit": "seconds"
        })

        if resp.get("success") and resp.get("spec"):
            spec = resp["spec"]
            # Robinson spec should have wide tolerance (50%)
            assert spec["nominal"] == 1.6
            assert abs(spec["tolerance_low"] - 0.8) < 0.01
            assert abs(spec["tolerance_high"] - 2.4) < 0.01
            assert spec["heat"] == 0.0  # No confidence yet
            print("✓ Robinson spec creation")
            return True
        print("✗ Robinson spec creation FAILED")
        return False

    async def test_measurement(self) -> bool:
        """Test: Measure value against spec"""
        # Measure in-spec value
        resp = await self.send({
            "type": "test_constraint",
            "action": "measure",
            "name": "test_duration",
            "value": 1.5
        })

        if resp.get("success"):
            result = resp["result"]
            assert result["in_spec"] == True
            assert result["R"] == 0.0  # No deviation
            assert result["polarity"] == 1
            print("✓ In-spec measurement")
        else:
            print("✗ In-spec measurement FAILED")
            return False

        # Measure out-of-spec value
        resp = await self.send({
            "type": "test_constraint",
            "action": "measure",
            "name": "test_duration",
            "value": 3.0  # Way over
        })

        if resp.get("success"):
            result = resp["result"]
            assert result["in_spec"] == False
            assert result["R"] > 0  # Has deviation
            assert result["polarity"] == -1
            print("✓ Out-of-spec measurement")
            return True
        print("✗ Out-of-spec measurement FAILED")
        return False

    async def test_verification(self) -> bool:
        """Test: Verify success adds heat"""
        # Get current heat
        resp = await self.send({
            "type": "test_constraint",
            "action": "get_spec",
            "name": "test_duration"
        })
        heat_before = resp["spec"]["heat"]

        # Verify success
        resp = await self.send({
            "type": "test_constraint",
            "action": "verify",
            "name": "test_duration",
            "value": 1.6,
            "success": True
        })

        if resp.get("success"):
            heat_added = resp.get("heat_added", 0)
            assert heat_added > 0  # K was added
            print(f"✓ Verification adds heat (+{heat_added:.3f})")
            return True
        print("✗ Verification FAILED")
        return False

    async def test_confidence_threshold(self) -> bool:
        """Test: After 5 verifications, should_exploit = True"""
        # Verify 5 times (need 5K heat for 5/6 confidence)
        for i in range(5):
            await self.send({
                "type": "test_constraint",
                "action": "verify",
                "name": "test_duration",
                "value": 1.6,
                "success": True
            })

        # Check confidence
        resp = await self.send({
            "type": "test_constraint",
            "action": "measure",
            "name": "test_duration",
            "value": 1.6
        })

        if resp.get("success"):
            result = resp["result"]
            if result["should_exploit"]:
                print(f"✓ Confidence threshold (heat={result['heat']:.2f}, exploit=True)")
                return True
            else:
                print(f"✗ Should exploit but doesn't (heat={result['heat']:.2f})")
                return False
        return False

    async def test_timed_input(self, duration: float, name: str) -> bool:
        """Test: Send timed input, verify spec learns"""
        print(f"  Testing {name} ({duration}s)...")

        # Press
        start = time.time()
        await self.send({
            "type": "input",
            "action": "press",
            "button": name
        })

        # Wait
        await asyncio.sleep(duration)

        # Release
        elapsed = time.time() - start
        resp = await self.send({
            "type": "input",
            "action": "release",
            "button": name,
            "duration": elapsed
        })

        if resp.get("success"):
            print(f"  ✓ {name}: {elapsed:.3f}s recorded")
            return True
        print(f"  ✗ {name}: FAILED")
        return False

    async def run_all_tests(self):
        """Run full diagnostic suite"""
        print()
        print("═" * 50)
        print("PBAI CONSTRAINT DIAGNOSTIC")
        print("═" * 50)
        print()

        results = []

        # Connect
        try:
            await self.connect()
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return

        try:
            # Sync clock
            results.append(("Clock sync", await self.sync_clock()))

            # Constraint tests
            results.append(("Robinson spec", await self.test_spec_creation()))
            results.append(("Measurement", await self.test_measurement()))
            results.append(("Verification", await self.test_verification()))
            results.append(("Confidence", await self.test_confidence_threshold()))

            # Timed input tests (like controller test)
            print()
            print("─" * 50)
            print("TIMED INPUT TESTS")
            print("─" * 50)

            test_buttons = [
                ("short_press", 0.1),
                ("medium_press", 0.5),
                ("long_press", 1.0),
                ("hold", 2.0),
            ]

            for name, duration in test_buttons:
                results.append((f"Input: {name}", await self.test_timed_input(duration, name)))

            # Summary
            print()
            print("═" * 50)
            print("RESULTS")
            print("═" * 50)

            passed = sum(1 for _, r in results if r)
            total = len(results)

            for name, result in results:
                status = "✓ PASS" if result else "✗ FAIL"
                print(f"  {name:30} {status}")

            print()
            print(f"  {passed}/{total} tests passed")
            print("═" * 50)

        finally:
            await self.disconnect()


async def main():
    if not HAS_WEBSOCKETS:
        print("Error: websockets package required")
        print("  pip install websockets")
        return
    
    client = DiagnosticClient()
    await client.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
