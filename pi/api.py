"""
PBAI Network API - Remote control interface

Simple REST API for controlling PBAI from anywhere on the network.

ENDPOINTS:
    GET  /status          - Get daemon status
    GET  /psychology      - Get psychology state
    GET  /manifold        - Get manifold summary
    POST /pause           - Pause daemon
    POST /resume          - Resume daemon
    POST /save            - Force save
    POST /inject/heat     - Inject heat into psychology
    POST /inject/perception - Inject a perception
    POST /environment/register - Register new environment
    GET  /environments    - List environments
"""

import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .daemon import PBAIDaemon

logger = logging.getLogger(__name__)


class APIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for PBAI API."""
    
    daemon: 'PBAIDaemon' = None  # Set by server
    
    def log_message(self, format, *args):
        """Override to use our logger."""
        logger.debug(f"API: {args[0]}")
    
    def _send_json(self, data: dict, status: int = 200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _send_error(self, message: str, status: int = 400):
        """Send error response."""
        self._send_json({"error": message}, status)
    
    def _read_body(self) -> dict:
        """Read and parse JSON body."""
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            return {}
        body = self.rfile.read(content_length)
        return json.loads(body.decode())
    
    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        
        if path == '/status':
            self._send_json(self.daemon.get_status())
        
        elif path == '/psychology':
            if not self.daemon.manifold:
                self._send_error("Manifold not initialized", 503)
                return
            
            m = self.daemon.manifold
            self._send_json({
                "identity": {
                    "heat": m.identity_node.heat if m.identity_node else 0,
                    "existence": m.identity_node.existence if m.identity_node else None,
                    "axes": len(m.identity_node.frame.axes) if m.identity_node else 0,
                },
                "ego": {
                    "heat": m.ego_node.heat if m.ego_node else 0,
                    "existence": m.ego_node.existence if m.ego_node else None,
                    "axes": len(m.ego_node.frame.axes) if m.ego_node else 0,
                },
                "conscience": {
                    "heat": m.conscience_node.heat if m.conscience_node else 0,
                    "existence": m.conscience_node.existence if m.conscience_node else None,
                    "axes": len(m.conscience_node.frame.axes) if m.conscience_node else 0,
                },
                "exploration_rate": m.get_exploration_rate(),
            })
        
        elif path == '/manifold':
            if not self.daemon.manifold:
                self._send_error("Manifold not initialized", 503)
                return
            
            m = self.daemon.manifold
            self._send_json({
                "nodes": len(m.nodes),
                "loop_number": m.loop_number,
                "total_heat": m.total_heat(),
                "node_list": [
                    {
                        "concept": n.concept,
                        "theta": n.theta,
                        "phi": n.phi,
                        "radius": n.radius,
                        "heat": n.heat,
                        "axes": len(n.frame.axes)
                    }
                    for n in list(m.nodes.values())[:50]  # First 50
                ]
            })
        
        elif path == '/environments':
            envs = {}
            if self.daemon.chooser:
                for k, v in self.daemon.chooser.stats.items():
                    envs[k] = {
                        "sessions": v.sessions,
                        "success_rate": v.success_rate,
                        "net_heat": v.net_heat,
                        "last_used": v.last_used.isoformat() if v.last_used else None
                    }
            self._send_json({"environments": envs})
        
        elif path == '/thermal':
            if self.daemon.thermal:
                state = self.daemon.thermal.update()
                self._send_json(state.to_dict())
            else:
                self._send_error("Thermal manager not initialized", 503)
        
        else:
            self._send_error(f"Unknown endpoint: {path}", 404)
    
    def do_POST(self):
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        
        if path == '/pause':
            self.daemon.pause()
            self._send_json({"status": "paused"})
        
        elif path == '/resume':
            self.daemon.resume()
            self._send_json({"status": "resumed"})
        
        elif path == '/save':
            self.daemon.force_save()
            self._send_json({"status": "saved"})
        
        elif path == '/inject/heat':
            try:
                body = self._read_body()
                amount = float(body.get('amount', 0))
                target = body.get('target', 'identity')
                self.daemon.inject_heat(amount, target)
                self._send_json({
                    "status": "injected",
                    "amount": amount,
                    "target": target
                })
            except Exception as e:
                self._send_error(str(e))
        
        elif path == '/inject/perception':
            try:
                body = self._read_body()
                self.daemon.inject_perception(body)
                self._send_json({"status": "perception_injected"})
            except Exception as e:
                self._send_error(str(e))
        
        elif path == '/action':
            try:
                body = self._read_body()
                action_type = body.get('action_type', 'observe')
                target = body.get('target')
                parameters = body.get('parameters', {})
                
                from drivers import Action
                action = Action(
                    action_type=action_type,
                    target=target,
                    parameters=parameters
                )
                result = self.daemon.env_core.act(action)
                self._send_json({
                    "success": result.success,
                    "outcome": result.outcome,
                    "heat_value": result.heat_value
                })
            except Exception as e:
                self._send_error(str(e))
        
        elif path == '/request_world':
            # Try buffer first (instant, no round-trip to PC)
            if self.daemon.body_server:
                latest = self.daemon.body_server.get_latest_world()
                if latest:
                    self._send_json({"status": "ok", "world": latest, "source": "buffer"})
                    return

            # Fall back to explicit request
            if not self.daemon.body_server or not self.daemon.body_server.has_body:
                self._send_error("No body connected", 503)
                return
            try:
                world = self.daemon.body_server.request_world(timeout=60.0)
                if world:
                    self._send_json({"status": "ok", "world": world, "source": "request"})
                else:
                    self._send_error("World request timed out or failed", 504)
            except Exception as e:
                self._send_error(str(e))

        elif path == '/body/action':
            # Send raw motor command to connected PC body
            if not self.daemon.body_server or not self.daemon.body_server.has_body:
                self._send_error("No body connected", 503)
                return
            try:
                body = self._read_body()
                success = self.daemon.body_server.send_action(body, timeout=5.0)
                self._send_json({"status": "ok", "sent": success})
            except Exception as e:
                self._send_error(str(e))

        elif path == '/choose':
            # Force environment choice
            if self.daemon.chooser:
                chosen = self.daemon.chooser.choose()
                self._send_json({"chosen": chosen})
            else:
                self._send_error("Chooser not initialized", 503)
        
        else:
            self._send_error(f"Unknown endpoint: {path}", 404)


class PBAIAPIServer(HTTPServer):
    """HTTP server for PBAI API."""
    
    def __init__(self, daemon: 'PBAIDaemon', port: int):
        self.daemon = daemon
        
        # Create handler class with daemon reference
        handler = type('Handler', (APIHandler,), {'daemon': daemon})
        
        super().__init__(('0.0.0.0', port), handler)


def create_api_server(daemon: 'PBAIDaemon', port: int) -> PBAIAPIServer:
    """Create API server."""
    return PBAIAPIServer(daemon, port)


def run_api_server(daemon: 'PBAIDaemon', port: int):
    """Run API server (blocking)."""
    server = create_api_server(daemon, port)
    logger.info(f"API server listening on 0.0.0.0:{port}")
    server.serve_forever()


# ═══════════════════════════════════════════════════════════════════════════════
# CLIENT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

class PBAIClient:
    """
    Client for connecting to PBAI daemon remotely.
    
    Usage:
        client = PBAIClient("192.168.1.100", 8420)
        status = client.get_status()
        client.inject_heat(1.0, "ego")
    """
    
    def __init__(self, host: str, port: int = 8420):
        self.base_url = f"http://{host}:{port}"
    
    def _get(self, path: str) -> dict:
        """Make GET request."""
        import urllib.request
        url = f"{self.base_url}{path}"
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read().decode())
    
    def _post(self, path: str, data: dict = None) -> dict:
        """Make POST request."""
        import urllib.request
        url = f"{self.base_url}{path}"
        body = json.dumps(data or {}).encode()
        req = urllib.request.Request(url, data=body, method='POST')
        req.add_header('Content-Type', 'application/json')
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())
    
    def get_status(self) -> dict:
        """Get daemon status."""
        return self._get('/status')
    
    def get_psychology(self) -> dict:
        """Get psychology state."""
        return self._get('/psychology')
    
    def get_manifold(self) -> dict:
        """Get manifold summary."""
        return self._get('/manifold')
    
    def get_environments(self) -> dict:
        """Get environment list."""
        return self._get('/environments')
    
    def get_thermal(self) -> dict:
        """Get thermal state."""
        return self._get('/thermal')
    
    def pause(self) -> dict:
        """Pause daemon."""
        return self._post('/pause')
    
    def resume(self) -> dict:
        """Resume daemon."""
        return self._post('/resume')
    
    def save(self) -> dict:
        """Force save."""
        return self._post('/save')
    
    def inject_heat(self, amount: float, target: str = "identity") -> dict:
        """Inject heat into psychology."""
        return self._post('/inject/heat', {"amount": amount, "target": target})
    
    def inject_perception(self, entities: list = None, events: list = None,
                          properties: dict = None) -> dict:
        """Inject a perception."""
        return self._post('/inject/perception', {
            "entities": entities or [],
            "events": events or [],
            "properties": properties or {}
        })
    
    def action(self, action_type: str, target: str = None, 
               parameters: dict = None) -> dict:
        """Execute an action."""
        return self._post('/action', {
            "action_type": action_type,
            "target": target,
            "parameters": parameters or {}
        })
    
    def choose(self) -> dict:
        """Force environment choice."""
        return self._post('/choose')
