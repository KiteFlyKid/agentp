import json
import os
import sys
import argparse
import subprocess
from typing import Optional, Dict, Any
import networkx as nx
from mitmproxy import http, ctx
from mitmproxy.tools.main import mitmdump
import logging
import re
from typing import Optional, List

# Add path for UTG imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.utg import utg
from agentp import (
    get_current_activity,
get_current_package,
    initialize_planner,
    generate_utg_description_with_plan,
)


class UTGPlanningAddon:
    """
    mitmproxy addon that intercepts requests to OPENAI API and adds UTG planning.
    Dynamically detects running app and loads corresponding UTG.
    """
    
    def __init__(self, utg_dir):
        self.utg_dir = utg_dir
        
        # UTG storage: package_name -> utg_instance
        self.utg_instances = {}
        self.utg_dataframes = {}
        
        # Current app state
        self.current_package = None
        self.current_utg_instance = None
        self.current_utg_df = None
        self.previous_activity = None
        self.current_activity = None
        
        # Initialize
        self._load_all_utgs()
        initialize_planner()
        print('finishing set up')
        print(f"UTG Planning Proxy initialized with {len(self.utg_instances)} UTGs loaded")
    
    def _parse_manifest(self, manifest_path: str) -> Optional[str]:
        """Parse manifest file to extract package name."""
        try:
            if not os.path.exists(manifest_path):
                return None
            
            with open(manifest_path, 'r') as f:
                content = f.read()
            
            # Look for package name in manifest
            # Format: "- package: com.dimowner.audiorecorder"
            package_match = re.search(r'- package:\s*([^\s\n]+)', content)
            if package_match:
                return package_match.group(1).strip()
            
            return None
        except Exception as e:
            ctx.log.error(f"Error parsing manifest {manifest_path}: {e}")
            return None
    
    def _load_all_utgs(self):
        """Load all UTGs from the UTG directory."""
        if not os.path.exists(self.utg_dir):
            ctx.log.error(f"UTG directory not found: {self.utg_dir}")
            return
        
        # Iterate through all subdirectories in utg_dir
        for app_dir in os.listdir(self.utg_dir):
            app_path = os.path.join(self.utg_dir, app_dir)
            
            if not os.path.isdir(app_path):
                continue
            
            # Parse manifest to get package name
            manifest_path = os.path.join(app_path, 'ManifestInfo', 'AndroidManifest.txt')
            package_name = self._parse_manifest(manifest_path)
            
            if not package_name:
                ctx.log.warn(f"Could not extract package name from {app_dir}, skipping")
                continue
            
            # Load UTG for this app
            xml_file = os.path.join(app_path, 'CTGResult', 'CTG.xml')
            component_info_file = os.path.join(app_path, 'CTGResult', 'componentInfo.xml')
            
            if not os.path.exists(xml_file):
                ctx.log.warn(f"UTG file not found for {app_dir}, skipping")
                continue
            
            try:
                # Load UTG instance
                if os.path.exists(component_info_file):
                    utg_instance = utg(
                        xml_file=xml_file,
                        component_info_file=component_info_file
                    )
                else:
                    utg_instance = utg(xml_file=xml_file)
                
                # Store UTG instance and dataframe by package name
                self.utg_instances[package_name] = utg_instance
                self.utg_dataframes[package_name] = nx.to_pandas_edgelist(utg_instance.graph)
                
                print(f"✓ Loaded UTG for {package_name} ({app_dir})")
                print(f"  - Edges: {len(self.utg_dataframes[package_name])}")
                print(f"  - Main activity: {utg_instance.main_activity}")
                
            except Exception as e:
                ctx.log.error(f"Error loading UTG for {app_dir}: {e}")

    def _switch_to_app(self, package_name: str) -> bool:
        """Switch to a different app's UTG context."""
        if package_name not in self.utg_instances:
            ctx.log.warn(f"No UTG loaded for package: {package_name}")
            return False
        
        if self.current_package != package_name:
            print(f"Switching context to app: {package_name}")
            self.current_package = package_name
            self.current_utg_instance = self.utg_instances[package_name]
            self.current_utg_df = self.utg_dataframes[package_name]
            self.previous_activity = None
            self.current_activity = None
        
        return True
    
    def _update_utg(self):
        """Update UTG with current activity information."""
        if not self.current_utg_instance:
            return
        
        try:
            new_activity = get_current_activity()
            
            if new_activity and self.previous_activity and new_activity != self.previous_activity:
                print(f"Activity transition: {self.previous_activity} -> {new_activity}")
                
                # Update UTG
                self.current_utg_instance.set_current_node(self.previous_activity)
                updated = self.current_utg_instance.dynamic_update_graph(new_node_name=new_activity)
                
                if updated:
                    print(f"✓ Added new edge/node to UTG")
                
                self.current_utg_instance.set_current_node(new_activity)
                self.current_utg_df = nx.to_pandas_edgelist(self.current_utg_instance.graph)
                
                # Update stored dataframe
                self.utg_dataframes[self.current_package] = self.current_utg_df
            
            self.previous_activity = self.current_activity
            self.current_activity = new_activity
            
        except Exception as e:
            ctx.log.error(f"Error updating UTG: {e}")
    


    def _extract_goal_from_messages(self, messages: List) -> Optional[str]:
        """Extract goal from chat messages."""
        for message in messages:
            if isinstance(message, dict) and message.get('role') == 'user':
                content = message.get('content', '')

                # Look for goal patterns (new prompt first)
                patterns = [
                    # Matches markdown header "**Current Request:**" with goal on same or next line(s),
                    # and stops right before the next bold section (e.g., "**Is the precondition met?**") or end of text.
                    r"(?:^|\n)\s*\*\*Current\s+Request:\*\*\s*(?:\n)?\s*(.+?)(?=\n\s*\*\*[^*\n]+?\*\*|\Z)",

                    # Flexible fallback: "Current Request ..." without bold
                    r"(?:^|\n)\s*Current\s+Request\s*:?\s*(.+?)(?=\n\s*\*\*[^*\n]+?\*\*|\Z)",

                    # Your older formats
                    r"(?:^|\n)\s*user\s+goal/request\s+is:\s*(.+?)(?:\n|$)",
                    r"(?:^|\n)\s*Goal:\s*(.+?)(?:\n|$)",
                    r"(?:^|\n)\s*Task:\s*(.+?)(?:\n|$)",
                ]

                for pattern in patterns:
                    m = re.search(pattern, content, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
                    if m:
                        goal = m.group(1).strip()
                        # Light cleanup: collapse internal extra whitespace
                        goal = re.sub(r"[ \t]+\n", "\n", goal)
                        goal = re.sub(r"\n{3,}", "\n\n", goal)
                        return goal

        return "Navigate the app"
    
    def _process_messages(self, messages: list) -> list:
        """Process messages and replace <AGENTP> tags."""
        modified_messages = []
        modified = False
        
        for message in messages:
            if isinstance(message, dict) and 'content' in message:
                content = message['content']
                
                if '<AGENTP>' in content:
                    # Update UTG state
                    self._update_utg()
                    
                    # Get current activity
                    current_activity = self.current_activity
                    if not current_activity:
                        ctx.log.warn("Could not get current activity")
                        utg_guide = '\nNo UTG navigation data available.\n'
                    else:
                        # Extract goal
                        goal_text = self._extract_goal_from_messages(messages)
                        
                        # Generate UTG navigation guide
                        try:
                            utg_guide = generate_utg_description_with_plan(
                                self.current_utg_df,
                                current_activity,
                                goal_text=goal_text,
                                utg_instance=self.current_utg_instance
                            )
                            print(f"Generated UTG guide for: {current_activity}")
                        except Exception as e:
                            ctx.log.error(f"Error generating UTG guide: {e}")
                            utg_guide = f'\nError generating navigation guide: {str(e)}\n'
                    
                    # Replace tag
                    content = content.replace('<AGENTP>', utg_guide)
                    modified = True
                
                # Create modified message
                modified_message = message.copy()
                modified_message['content'] = content
                modified_messages.append(modified_message)
            else:
                modified_messages.append(message)
        
        if modified:
            print("✓ Modified prompt with UTG planning information")
        
        return modified_messages
    
    def request(self, flow: http.HTTPFlow) -> None:
        """
        Intercept and modify requests to the OPENAI API.
        """
        # Check if this is a request to the target API
        if "api.openai.com" in flow.request.pretty_host:
            print(f"Intercepted request to: {flow.request.pretty_url}")
            
            # Check if it's a chat completions request
            if "/chat/completions" in flow.request.path:
                try:
                    # Get current running app package
                    current_package = get_current_package()
                    
                    if not current_package:
                        ctx.log.warn("Could not detect current running app")
                        return
                    
                    print(f"Current app: {current_package}")
                    
                    # Check if we have UTG for this app
                    if current_package not in self.utg_instances:
                        print(f"No UTG available for {current_package}, passing through")
                        return
                    
                    # Switch to this app's UTG context
                    if not self._switch_to_app(current_package):
                        return
                    
                    # Parse request body
                    request_data = json.loads(flow.request.content)
                    
                    # Check if messages contain <AGENTP> tag
                    if 'messages' in request_data:
                        # Process messages with the current app's UTG
                        modified_messages = self._process_messages(request_data['messages'])
                        
                        # Update request data
                        request_data['messages'] = modified_messages
                        
                        # Update request content
                        flow.request.content = json.dumps(request_data).encode('utf-8')
                        
                except Exception as e:
                    ctx.log.error(f"Error processing request: {e}")
    
    def response(self, flow: http.HTTPFlow) -> None:
        """
        Log responses for debugging.
        """
        if "api.openai.com" in flow.request.pretty_host:
            if flow.response and flow.response.status_code == 200:
                print("✓ Response received successfully")
            else:
                ctx.log.error(f"Response error: {flow.response.status_code if flow.response else 'No response'}")
    
    def get_status(self) -> str:
        """Get current status of the addon."""
        status = f"UTG Planning Proxy Status:\n"
        status += f"- Loaded UTGs: {len(self.utg_instances)}\n"
        status += f"- Available packages:\n"
        for pkg in self.utg_instances.keys():
            status += f"  • {pkg}\n"
        status += f"- Current app: {self.current_package or 'None'}\n"
        return status


addon_instance = None  # keep the global

def load(loader):
    """Called by mitmproxy at startup to register options."""
    loader.add_option("utg_dir", str, "ICCBot_output", 
                      "UTG directory containing app folders")

def configure(updated):
    """
    Called by mitmproxy when options change (incl. initial parse).
    Create/remove the addon dynamically based on presence of options.
    """
    global addon_instance

    # If options changed and an addon exists, remove it first.
    if addon_instance is not None and "utg_dir" in updated:
        try:
            ctx.master.addons.remove(addon_instance)
        except Exception as e:
            ctx.log.warn(f"Failed removing previous addon: {e}")
        addon_instance = None

    # Create addon with UTG directory
    if addon_instance is None:
        addon_instance = UTGPlanningAddon(ctx.options.utg_dir)
        ctx.master.addons.add(addon_instance)
        print(addon_instance.get_status())

# Export the addon properly
addons = [lambda: addon_instance] if addon_instance else []


# For direct Python execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UTG Planning Network Proxy')
    parser.add_argument('--utg_dir',
                       help='UTG directory containing app folders')
    parser.add_argument('--port', type=int, default=8080, help='Proxy port (default: 8080)')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    
    args = parser.parse_args()
    print(f"""
╔════════════════════════════════════════════════════╗
║     UTG Planning Network Proxy (Dynamic Mode)      ║
╠════════════════════════════════════════════════════╣
║ UTG Dir: {args.utg_dir:<42}║
║ Port:    {args.port:<42}║
╠════════════════════════════════════════════════════╣
║ Configure your agent to use proxy:                 ║
║ HTTP_PROXY=http://localhost:{args.port:<22}║
║ HTTPS_PROXY=http://localhost:{args.port:<21}║
╠════════════════════════════════════════════════════╣
║ Features:                                           ║
║ • Auto-detects running Android app                 ║
║ • Loads UTG if available for the app               ║
║ • Add <AGENTP> in prompts for planning info        ║
╚════════════════════════════════════════════════════╝
    
Loading UTGs from directory...
    """)
    
    # Start mitmdump with our script
    from mitmproxy.tools.main import mitmdump
    
    # Build command line arguments for mitmdump
    mitmdump_args = [
        '--listen-host', args.host,  # Listen on all interfaces
        '--listen-port', str(args.port),
        '-s', __file__,
        '--set', f'confdir=~/.mitmproxy',
        '--set', f'utg_dir={utg_dir}',
        '--set', 'ssl_insecure=true',  # Accept untrusted upstream certs
        '--verbose',
    ]
    
    # Run mitmdump
    mitmdump(mitmdump_args)