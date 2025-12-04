import json
import os
import subprocess
import pandas as pd
import networkx as nx
from utils.utg import utg  # Assuming this is your UTG module
from typing import List, Tuple, Optional, Dict
import tempfile
import re
from dataclasses import dataclass
from openai import OpenAI
import openai

openai.verify_ssl_certs = False

@dataclass
class Goal:
    """Represents a user goal with type and target nodes"""
    goal_text: str
    goal_type: str
    target_nodes: List[str] = None
    confidence: float = 0.0
    llm_raw_response: str = None

    def __post_init__(self):
        if self.target_nodes is None:
            self.target_nodes = []


@dataclass
class PlanStep:
    """Represents a single step in a plan"""
    action: str
    from_node: str
    to_node: str

    def __str__(self):
        return f"Navigate from {self.from_node} to {self.to_node}"


class TargetNodeSelector:
    """Module for selecting target nodes from UTG based on user goals."""

    def __init__(self, llm_base_url: str = "https://api.openai.com/v1"):
        self.llm_client = None
        if 'OPENAI_API_KEY' not in os.environ:
            try:
                with open("key", 'r') as f:
                    api_key = f.read().strip()
            except FileNotFoundError:
                raise RuntimeError(
                    'OPENAI API key not found. Set OPENAI_API_KEY environment variable or create "key" file.')
        else:
            api_key = os.environ['OPENAI_API_KEY']

        if api_key:
            self.llm_client = OpenAI(
                base_url=llm_base_url,
                api_key=api_key,
            )
    def _re_select_nodes(self, user_goal: str, utg) -> Tuple[List[str], float, Dict]:
        pass
    def select_target_for_goal(self, goal_text: str, utg) -> Tuple[List[str], float]:
        """
        Select target nodes based on goal text using LLM or fuzzy matching.
        Returns list of target nodes and confidence.
        """
        if self.llm_client:
            target_nodes, confidence, _ = self._llm_select_nodes(goal_text, utg)
            return target_nodes, confidence
        else:
            # Fallback to keyword matching
            return self._fuzzy_match(goal_text, utg)

    def _fuzzy_match(self, user_goal: str, utg) -> Tuple[List[str], float]:
        """Perform fuzzy matching based on keywords."""
        nodes = utg.get_nodes()
        node_scores = []

        # Extract keywords from the goal
        keywords = self._extract_keywords(user_goal)

        for node in nodes:
            score = self._calculate_match_score(node, keywords)
            if score > 0:
                node_scores.append((node, score))

        if node_scores:
            node_scores.sort(key=lambda x: x[1], reverse=True)
            top_matches = [node for node, score in node_scores[:3] if score > 0.3]
            if top_matches:
                return top_matches, node_scores[0][1]

        return [], 0.0

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
                      'is', 'are', 'was', 'were', 'using', 'app'}
        words = text.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return keywords

    def _calculate_match_score(self, node: str, keywords: List[str]) -> float:
        """Calculate matching score between node name and keywords."""
        node_lower = node.lower()
        score = 0.0

        for keyword in keywords:
            if keyword in node_lower:
                score += 1.0
            elif any(keyword in part for part in node_lower.split('.')):
                score += 0.5

        return score / max(len(keywords), 1)

    def _llm_select_nodes(self, user_goal: str, utg) -> Tuple[List[str], float, Dict]:
        """Use LLM to select appropriate nodes based on natural language goal."""
        if not self.llm_client:
            return [], 0.0, {}

        nodes_ctx = []
        for node in utg.get_nodes():
            node_type = utg.graph.nodes[node].get('type', 'Unknown')
            nodes_ctx.append(f"{node} (Type: {node_type})")

        prompt = f"""Given the user goal: "{user_goal}".

Select the most relevant nodes to achieve this goal.

Available nodes:
{chr(10).join(nodes_ctx)}

Return your response in JSON format:
{{
    "nodes": ["node1", "node2", ...],
    "confidence": 0.8,
    "reasoning": "brief explanation"
}}

Return ONLY the JSON, no other text."""

        try:
            completion = self.llm_client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "You are an expert in Android app navigation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )

            response_text = completion.choices[0].message.content.strip()

            # Parse JSON response
            if response_text.strip().startswith('```'):
                lines = response_text.strip().split('\n')
                json_lines = []
                in_json = False
                for line in lines:
                    if line.strip().startswith('```json'):
                        in_json = True
                        continue
                    elif line.strip() == '```':
                        break
                    elif in_json:
                        json_lines.append(line)
                response_text = '\n'.join(json_lines)

            response_data = json.loads(response_text.strip())
            nodes = response_data.get("nodes", [])
            confidence = response_data.get("confidence", 0.7)

            # Validate nodes exist in UTG
            valid_nodes = []
            all_nodes = utg.get_nodes()
            for node in nodes:
                if node in all_nodes:
                    valid_nodes.append(node)

            return valid_nodes, confidence, response_data

        except Exception as e:
            print(f"Error calling LLM: {e}")
            return [], 0.0, {}


class PDDLGenerator:
    """Module for generating PDDL domain and problem files from UTG."""

    def __init__(self, planner_path: str = "downward/fast-downward.py"):
        """
        Initialize the PDDL generator.

        Args:
            planner_path: Path to the fast-downward.py script
        """
        self.planner_path = planner_path
        self.node_name_mapping = {}

    def generate_pddl(self, utg, target_nodes: List[str], current_node: Optional[str] = None) -> Tuple[str, str]:
        """
        Generate PDDL domain and problem files from UTG with multiple target nodes.

        Args:
            utg: UTG instance
            target_nodes: List of target nodes to reach
            current_node: Current node (defaults to main_activity)

        Returns:
            Tuple of (domain_pddl, problem_pddl) as strings
        """
        if current_node is None:
            current_node = utg.main_activity

        # Generate domain
        domain_pddl = self._generate_domain()

        # Generate problem
        problem_pddl = self._generate_problem(utg, current_node, target_nodes)

        return domain_pddl, problem_pddl

    def _generate_domain(self) -> str:
        """Return the PDDL domain with support for multiple goals."""
        return """(define (domain android-utg-navigation)
  (:requirements :strips :typing)

  (:types
    node - object
  )

  (:predicates
    (at ?n - node)
    (connected ?from - node ?to - node)
    (visited ?n - node)
    (goal-node ?n - node)
    (goal-achieved ?n - node)
  )

  (:action navigate
    :parameters (?from - node ?to - node)
    :precondition (and 
      (at ?from)
      (connected ?from ?to)
    )
    :effect (and 
      (not (at ?from))
      (at ?to)
      (visited ?to)
      (when (goal-node ?to) (goal-achieved ?to))
    )
  )
)"""

    def _generate_problem(self, utg, start_node: str, target_nodes: List[str]) -> str:
        """Generate problem PDDL from UTG with multiple target nodes."""
        # Get all nodes
        nodes = utg.get_nodes()

        # Get all edges
        edges = utg.graph.edges()

        # Create safe node names for PDDL (replace special characters)
        safe_nodes = {}
        for node in nodes:
            safe_name = self._make_safe_name(node)
            safe_nodes[node] = safe_name

        # Build reverse mapping
        self.node_name_mapping = {v: k for k, v in safe_nodes.items()}

        # Generate objects section
        objects = "\n    ".join([f"{safe_name} - node" for safe_name in safe_nodes.values()])

        # Generate init section
        init_predicates = [f"(at {safe_nodes[start_node]})"]

        # Add goal nodes
        for target in target_nodes:
            if target in safe_nodes:
                init_predicates.append(f"(goal-node {safe_nodes[target]})")

        # Add all connections
        for from_node, to_node in edges:
            if from_node in safe_nodes and to_node in safe_nodes:
                init_predicates.append(f"(connected {safe_nodes[from_node]} {safe_nodes[to_node]})")

        init = "\n    ".join(init_predicates)

        # Generate goal - all target nodes must be achieved
        goal_conditions = []
        for target in target_nodes:
            if target in safe_nodes:
                goal_conditions.append(f"(goal-achieved {safe_nodes[target]})")

        if len(goal_conditions) == 1:
            goal = goal_conditions[0]
        else:
            goal = f"(and {' '.join(goal_conditions)})"

        # Construct problem
        problem_pddl = f"""(define (problem utg-navigation-problem)
  (:domain android-utg-navigation)

  (:objects
    {objects}
  )

  (:init
    {init}
  )

  (:goal
    {goal}
  )
)"""

        return problem_pddl

    def _make_safe_name(self, node_name: str) -> str:
        """Convert node name to PDDL-safe identifier."""
        # Replace special characters with underscores
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', node_name)
        # Ensure it starts with a letter
        if safe_name and not safe_name[0].isalpha():
            safe_name = 'n_' + safe_name
        return safe_name.lower()

    def solve_pddl(self, domain_pddl: str, problem_pddl: str) -> Optional[str]:
        """
        Solve PDDL using Fast Downward planner.

        Args:
            domain_pddl: Domain PDDL as string
            problem_pddl: Problem PDDL as string

        Returns:
            Plan in SAS format or None if no solution
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write PDDL files
            domain_file = os.path.join(temp_dir, "domain.pddl")
            problem_file = os.path.join(temp_dir, "problem.pddl")

            with open(domain_file, 'w') as f:
                f.write(domain_pddl)

            with open(problem_file, 'w') as f:
                f.write(problem_pddl)

            # Run planner
            plan_file = os.path.join(temp_dir, "sas_plan")

            try:
                # Fast Downward command using Python script
                cmd = [
                    "python",
                    self.planner_path,
                    domain_file,
                    problem_file,
                    "--search", "astar(ff())"
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)

                # Check if solution found
                possible_plan_files = [
                    plan_file,
                    os.path.join(temp_dir, "sas_plan"),
                    os.path.join(os.path.dirname(self.planner_path), "sas_plan"),
                    "sas_plan"
                ]

                for pf in possible_plan_files:
                    if os.path.exists(pf):
                        with open(pf, 'r') as f:
                            plan_content = f.read()
                        # Clean up the plan file if it's in the planner directory
                        if pf in [os.path.join(os.path.dirname(self.planner_path), "sas_plan"), "sas_plan"]:
                            try:
                                os.remove(pf)
                            except:
                                pass
                        return plan_content

                # If no plan file found, check stdout for plan or errors
                if "Solution found" in result.stdout:
                    lines = result.stdout.split('\n')
                    plan_lines = []
                    in_plan = False
                    for line in lines:
                        if line.strip().startswith('(') and line.strip().endswith(')'):
                            in_plan = True
                            plan_lines.append(line.strip())
                        elif in_plan and not line.strip():
                            break
                    if plan_lines:
                        return '\n'.join(plan_lines)

                # Print error information for debugging
                if result.stderr:
                    print(f"Planner stderr: {result.stderr}")
                if "Solution found" not in result.stdout and result.stdout:
                    print(f"Planner stdout: {result.stdout[:500]}")

            except Exception as e:
                print(f"Error running planner: {e}")

        return None

    def parse_plan(self, sas_plan: str) -> List[PlanStep]:
        """
        Parse SAS plan into structured format.

        Args:
            sas_plan: Plan in SAS format

        Returns:
            List of PlanStep objects
        """
        plan_steps = []

        # Parse each line of the plan
        for line in sas_plan.strip().split('\n'):
            line = line.strip()
            if line.startswith('(') and line.endswith(')'):
                # Extract action and parameters
                match = re.match(r'\((\w+)\s+(\w+)\s+(\w+)\)', line)
                if match:
                    action, from_node, to_node = match.groups()
                    # Convert safe names back to original names
                    from_node_orig = self.node_name_mapping.get(from_node, from_node)
                    to_node_orig = self.node_name_mapping.get(to_node, to_node)
                    plan_steps.append(PlanStep(action, from_node_orig, to_node_orig))

        return plan_steps

    def find_plan(self, utg, current_node: str, target_nodes: List[str]) -> Optional[List[PlanStep]]:
        """
        Find a plan from current node to target nodes using PDDL planner.
        Returns None if no plan exists.

        Args:
            utg: UTG instance
            current_node: Current node
            target_nodes: List of target nodes

        Returns:
            List of PlanStep objects or None if no plan found
        """
        if not target_nodes:
            return None

        # First check basic reachability using NetworkX
        reachable_targets = []
        for target in target_nodes:
            if target==current_node:
                continue
            try:
                # Quick reachability check
                if nx.has_path(utg.graph, current_node, target):
                    reachable_targets.append(target)
            except:
                continue

        if not reachable_targets:
            return None

        # Generate PDDL files
        domain_pddl, problem_pddl = self.generate_pddl(utg, reachable_targets, current_node)

        # Solve using PDDL planner
        sas_plan = self.solve_pddl(domain_pddl, problem_pddl)

        if sas_plan:
            # Parse and return plan steps
            return self.parse_plan(sas_plan)

        return None


# Global instances for planner and target selector
_pddl_generator_instance = None
_target_selector_instance = None


def initialize_planner(planner_path: str = "downward/fast-downward.py"):
    """Initialize global planner and target selector instances."""
    global _pddl_generator_instance, _target_selector_instance
    _pddl_generator_instance = PDDLGenerator(planner_path=planner_path)
    _target_selector_instance = TargetNodeSelector()

def get_current_package():
    """Get the package name of the currently running app using ADB."""
    try:
        result = subprocess.run(
            ['adb', 'shell', 'dumpsys', 'activity', 'activities'],
            capture_output=True,
            text=True,
            check=True
        )

        lines = result.stdout.split('\n')
        for line in lines:
            if 'mResumedActivity' in line or 'mCurrentFocus' in line:
                if '/' in line:
                    # Extract the part before the slash (package name)
                    # Format: package.name/ActivityName
                    parts = line.split('/')
                    for part in parts:
                        if '.' in part:
                            # Find the package name pattern
                            match = re.search(r'([a-zA-Z0-9._]+\.[a-zA-Z0-9._]+)', part)
                            if match:
                                package_name = match.group(1)
                                # Clean up any trailing characters
                                package_name = package_name.split()[0].strip()
                                return package_name

        # Alternative method using dumpsys activity top
        result2 = subprocess.run(
            ['adb', 'shell', 'dumpsys', 'activity', 'top'],
            capture_output=True,
            text=True,
            check=True
        )

        lines2 = result2.stdout.split('\n')
        for line in lines2:
            if 'ACTIVITY' in line and '/' in line:
                # Extract package name (before the slash)
                parts = line.split('/')
                for part in parts:
                    if '.' in part:
                        match = re.search(r'([a-zA-Z0-9._]+\.[a-zA-Z0-9._]+)', part)
                        if match:
                            package_name = match.group(1)
                            package_name = package_name.split()[0].strip()
                            return package_name

        return None

    except Exception as e:
        print(f"Error getting current package: {e}")
        return None

def get_current_activity():
    """Get the current activity name using adb command."""
    try:
        result = subprocess.run(
            ['adb', 'shell', 'dumpsys', 'activity', 'activities'],
            capture_output=True,
            text=True,
            check=True
        )

        lines = result.stdout.split('\n')
        for line in lines:
            if 'mResumedActivity' in line or 'mCurrentFocus' in line:
                if '/' in line:
                    activity_info = line.split('/')[-1].split()[0]
                    current_activity=activity_info.strip()
                    current_activity_clean = str(current_activity).strip().replace('}', '')
                    return current_activity_clean

        # Alternative method
        result2 = subprocess.run(
            ['adb', 'shell', 'dumpsys', 'activity', 'top'],
            capture_output=True,
            text=True,
            check=True
        )

        lines2 = result2.stdout.split('\n')
        for line in lines2:
            if 'ACTIVITY' in line and '/' in line:
                activity_info = line.split('/')[-1].split()[0]
                current_activity = activity_info.strip()
                current_activity_clean = str(current_activity).strip().replace('}', '')
                return current_activity_clean

        return None

    except Exception as e:
        print(f"Error getting current activity: {e}")
        return None




def map_task_to_utg_app(task_name):
    """Map task name to UTG app name using the provided mappings."""

    app_to_utg_mapping = {
        'music': 'retro_music',
        'calendar': 'simple_calendar_pro',
        'audio recorder': 'audio_recorder',
        'osmand': 'osmand',
        'tasks': 'tasks',
        'recipe': 'broccoli_app',
        'sms': 'simple_sms_messenger',
        'simpledrawpro': 'simple_draw_pro',
        'markor': 'markor',
        'sportstracker': 'open_tracks_sports_tracker',
        'vlc': 'vlc',
        'gallery': 'simple_gallery_pro',
        'expense': 'pro_expense',
        'joplin': 'joplin',
    }

    task_to_app_mapping = {
        "Audio Recorder Record Audio": "audio recorder",
        "Audio Recorder Record Audio With File Name": "audio recorder",
        "Browser Draw": "files, chrome",
        "Browser Maze": "files, chrome",
        "Browser Multiply": "files, chrome",
        "Camera Take Photo": "camera",
        "Camera Take Video": "camera",
        "Clock Stop Watch Paused Verify": "clock",
        "Clock Stop Watch Running": "clock",
        "Clock Timer Entry": "clock",
        "Contacts Add Contact": "contacts",
        "Contacts New Contact Draft": "contacts",
        "Expense Add Multiple": "expense",
        "Expense Add Multiple From Gallery": "gallery, expense",
        "Expense Add Multiple From Markor": "markor, expense",
        "Expense Add Single": "expense",
        "Expense Delete Duplicates": "expense",
        "Expense Delete Duplicates2": "expense",
        "Expense Delete Multiple": "expense",
        "Expense Delete Multiple2": "expense",
        "Expense Delete Single": "expense",
        "Files Delete File": "files",
        "Files Move File": "files",
        "Markor Add Note Header": "markor",
        "Markor Change Note Content": "markor",
        "Markor Create Folder": "markor",
        "Markor Create Note": "markor",
        "Markor Create Note And Sms": "markor, sms",
        "Markor Create Note From Clipboard": "markor",
        "Markor Delete All Notes": "markor",
        "Markor Delete Newest Note": "markor",
        "Markor Delete Note": "markor",
        "Markor Edit Note": "markor",
        "Markor Merge Notes": "markor",
        "Markor Move Note": "markor",
        "Markor Transcribe Receipt": "gallery, markor",
        "Markor Transcribe Video": "markor, vlc",
        "Notes Is Todo": "joplin",
        "Notes Meeting Attendee Count": "joplin",
        "Notes Recipe Ingredient Count": "joplin",
        "Notes Todo Item Count": "joplin",
        "Open App Task Eval": "camera, clock, contacts, settings, dialer",
        "Osm And Favorite": "osmand",
        "Osm And Marker": "osmand",
        "Osm And Track": "osmand",
        "Recipe Add Multiple Recipes": "recipe",
        "Recipe Add Multiple Recipes From Image": "markor, recipe",
        "Recipe Add Multiple Recipes From Markor": "gallery, recipe",
        "Recipe Add Multiple Recipes From Markor2": "markor, recipe",
        "Recipe Add Single Recipe": "recipe",
        "Recipe Delete Duplicate Recipes": "recipe",
        "Recipe Delete Duplicate Recipes2": "recipe",
        "Recipe Delete Duplicate Recipes3": "recipe",
        "Recipe Delete Multiple Recipes": "recipe",
        "Recipe Delete Multiple Recipes With Constraint": "recipe",
        "Recipe Delete Multiple Recipes With Noise": "recipe",
        "Recipe Delete Single Recipe": "recipe",
        "Recipe Delete Single Recipe With Noise": "recipe",
        "Retro Create Playlist": "music",
        "Retro Playing Queue": "music",
        "Retro Playlist Duration": "music",
        "Retro Save Playlist": "music",
        "Save Copy Of Receipt Task Eval": "gallery",
        "Simple Calendar Add One Event": "calendar",
        "Simple Calendar Add One Event In Two Weeks": "calendar",
        "Simple Calendar Add One Event Relative Day": "calendar",
        "Simple Calendar Add One Event Tomorrow": "calendar",
        "Simple Calendar Add Repeating Event": "calendar",
        "Simple Calendar Any Events On Date": "calendar",
        "Simple Calendar Delete Events": "calendar",
        "Simple Calendar Delete Events On Relative Day": "calendar",
        "Simple Calendar Delete One Event": "calendar",
        "Simple Calendar Event On Date At Time": "calendar",
        "Simple Calendar Events In Next Week": "calendar",
        "Simple Calendar Events In Time Range": "calendar",
        "Simple Calendar Events On Date": "calendar",
        "Simple Calendar First Event After Start Time": "calendar",
        "Simple Calendar Location Of Event": "calendar",
        "Simple Calendar Next Event": "calendar",
        "Simple Calendar Next Meeting With Person": "calendar",
        "Simple Draw Pro Create Drawing": "simpledrawpro",
        "Simple Sms Reply": "sms",
        "Simple Sms Reply Most Recent": "sms",
        "Simple Sms Resend": "sms",
        "Simple Sms Send": "sms",
        "Simple Sms Send Clipboard Content": "sms",
        "Simple Sms Send Received Address": "sms",
        "Sports Tracker Activities Count For Week": "sportstracker",
        "Sports Tracker Activities On Date": "sportstracker",
        "Sports Tracker Activity Duration": "sportstracker",
        "Sports Tracker Longest Distance Activity": "sportstracker",
        "Sports Tracker Total Distance For Category Over Interval": "sportstracker",
        "Sports Tracker Total Duration For Category Over Interval": "sportstracker",
        "System Bluetooth Turn Off": "settings",
        "System Bluetooth Turn Off Verify": "settings",
        "System Bluetooth Turn On": "settings",
        "System Bluetooth Turn On Verify": "settings",
        "System Brightness Max": "settings",
        "System Brightness Max Verify": "settings",
        "System Brightness Min": "settings",
        "System Brightness Min Verify": "settings",
        "System Copy To Clipboard": "n/a",
        "System Wifi Turn Off": "settings",
        "System Wifi Turn Off Verify": "settings",
        "System Wifi Turn On": "settings",
        "System Wifi Turn On Verify": "settings",
        "Tasks Completed For Date": "tasks",
        "Tasks Due Next Week": "tasks",
        "Tasks Due On Date": "tasks",
        "Tasks High Priority Tasks": "tasks",
        "Tasks High Priority Tasks Due On Date": "tasks",
        "Tasks Incomplete Tasks On Date": "tasks",
        "Turn Off Wifi And Turn On Bluetooth": "settings",
        "Turn On Wifi And Open App": "settings",
        "Vlc Create Playlist": "vlc",
        "Vlc Create Two Playlists": "vlc"
    }

    if task_name not in task_to_app_mapping:
        raise ValueError(f"Task '{task_name}' not found in task mapping")

    app_names = task_to_app_mapping[task_name]

    if ',' in app_names:
        app_name = app_names.split(',')[0].strip()
    else:
        app_name = app_names.strip()

    if app_name not in app_to_utg_mapping:
        # Return the app name as is if not in mapping
        return app_name.replace(' ', '_')

    return app_to_utg_mapping[app_name]


def read_utg_for_app(utg_app_name, pure_utg=False, utg_dir='ICCBot_output'):
    """Read UTG for a specific app and return pure_utg_df."""
    app_path = os.path.join(utg_dir, utg_app_name)

    if not os.path.isdir(app_path):
        raise FileNotFoundError(f"UTG directory not found for app: {utg_app_name}")

    xml_file = os.path.join(app_path, 'CTGResult', 'CTG.xml')

    if not os.path.exists(xml_file):
        raise FileNotFoundError(f"CTG.xml not found for app: {utg_app_name}")

    try:
        print(f"Processing {utg_app_name}...")
        graph = utg(xml_file=xml_file)
        if pure_utg:
            utg_df = nx.to_pandas_edgelist(graph.pure_utg)
        else:
            utg_df = nx.to_pandas_edgelist(graph)
        print(f"✓ Successfully processed {utg_app_name} with {len(utg_df)} edges.")
        return utg_df
    except Exception as e:
        raise RuntimeError(f"Failed to process {utg_app_name}: {e}")


def get_utg_for_task(task_name, utg_dir='ICCBot_output'):
    """Get UTG for a specific task."""
    current_activity = get_current_activity()
    if current_activity:
        print(f"Current activity: {current_activity}")
    else:
        print("Warning: Could not determine current activity")

    try:
        utg_app_name = map_task_to_utg_app(task_name)
        print(f"Task '{task_name}' mapped to UTG app: '{utg_app_name}'")
    except ValueError as e:
        print(f"Error in mapping: {e}")
        return None

    try:
        utg_df = read_utg_for_app(utg_app_name, utg_dir)
        return utg_df
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error reading UTG: {e}")
        return None


def generate_utg_description_with_plan(pure_utg_df, current_activity, goal_text=None, utg_instance=None):
    """
    Enhanced UTG description generator that includes planning information using PDDL planner.

    Args:
        pure_utg_df: DataFrame containing the pure UTG data
        current_activity: Current activity name
        goal_text: Optional goal text for planning
        utg_instance: Optional UTG instance for planning

    Returns:
        str: Formatted UTG description with planning guidance
    """

    if pure_utg_df is None or pure_utg_df.empty:
        return "\nNo UTG navigation data available.\n"

    if not current_activity:
        return "\nCurrent activity not available, cannot provide specific UTG guidance.\n"

    # Start with basic navigation information
    utg_description = f"\n--- UTG Navigation Guide ---\n"
    utg_description += f"Current Activity: {current_activity}\n\n"

    # Clean current activity for matching


    # Filter dataframe for current activity
    informative_columns = ['source', 'target', 'action', 'unit', 'method', 'extras', 'type']
    available_columns = [col for col in informative_columns if col in pure_utg_df.columns]

    if not available_columns:
        return "\nUTG data contains no informative navigation information.\n"

    filtered_df = pure_utg_df[available_columns].copy()

    # Filter by current activity as source
    if 'source' in filtered_df.columns:
        activity_mask = (
                (filtered_df['source'].str.contains(current_activity, case=False, na=False)) |
                (filtered_df['source'].str.endswith(current_activity, na=False)) |
                (filtered_df['source'] == current_activity)
        )
        current_activity_df = filtered_df[activity_mask].copy()
    else:
        return "\nUTG data missing source information.\n"

    # Generate Planning Information using PDDL if goal and UTG instance provided

    if current_activity_df.empty:
        utg_description += f"No navigation options found from current activity.\n"
        utg_description += "This might be a new screen or you may need to go back.\n"
    else:
        # Group by target destination
        if 'target' in current_activity_df.columns:
            current_activity
            target_groups = current_activity_df.groupby('target')
            if len(target_groups)>1: # if there are multiple activities to navigate, then plan; or there will be only one choice no need to plan

                plan_section = ""
                if goal_text and utg_instance and _pddl_generator_instance and _target_selector_instance:
                    try:
                        # Get target nodes for the goal
                        target_nodes, confidence = _target_selector_instance.select_target_for_goal(
                            goal_text, utg_instance
                        )

                        if target_nodes and confidence > 0.3:
                            # Find plan using PDDL planner
                            plan_steps = _pddl_generator_instance.find_plan(
                                utg_instance, current_activity, target_nodes
                            )

                            if plan_steps:
                                plan_section = "\n--- NAVIGATION PLAN FOR YOUR GOAL ---\n"
                                plan_section += f"Goal Analysis: Based on your goal, the system identified these target destinations:\n"
                                for target in target_nodes:  # Show max 3 targets
                                    target_name = target.split('.')[-1] if '.' in target else target
                                    plan_section += f"  • {target_name}\n"
                                plan_section += f"Confidence: {confidence:.0%}\n\n"

                                plan_section += "OPTIMAL PATH (Follow these steps in order):\n"
                                for i, step in enumerate(plan_steps[:10], 1):  # Limit to 10 steps
                                    from_name = step.from_node.split('.')[-1] if '.' in step.from_node else step.from_node
                                    to_name = step.to_node.split('.')[-1] if '.' in step.to_node else step.to_node
                                    plan_section += f"Step {i}: Navigate from {from_name} to {to_name}\n"

                                if len(plan_steps) > 10:
                                    plan_section += f"... and {len(plan_steps) - 10} more steps\n"

                                # Find immediate next action from current screen
                                if plan_steps:
                                    next_target = plan_steps[0].to_node
                                    # Find the specific UI element to click
                                    next_actions = current_activity_df[current_activity_df['target'] == next_target]
                                    if not next_actions.empty:
                                        plan_section += "\n⚡ IMMEDIATE NEXT ACTION:\n"
                                        row = next_actions.iloc[0]
                                        action_desc = []
                                        if 'action' in row and pd.notna(row['action']):
                                            action_desc.append(f"{row['action'].upper()}")
                                        if 'unit' in row and pd.notna(row['unit']):
                                            action_desc.append(f"on '{row['unit']}'")
                                        if 'type' in row and pd.notna(row['type']):
                                            action_desc.append(f"({row['type']})")
                                        plan_section += f"  → {' '.join(action_desc)}\n"
                                        next_name = next_target.split('.')[-1] if '.' in next_target else next_target
                                        plan_section += f"  This will take you to: {next_name}\n"

                                plan_section += f"\n✅ Total steps in optimal path: {len(plan_steps)}\n"
                                plan_section += "💡 This path was computed using PDDL planning for guaranteed optimality\n\n"
                                utg_description += plan_section
                                utg_description += "\n--- USAGE TIPS ---\n"
                                utg_description += "• If a NAVIGATION PLAN is shown above, follow it step by step for optimal path\n"
                                utg_description += "• The plan was computed using formal PDDL planning algorithms\n"
                                utg_description += "• If no plan exists, the target is unreachable from current location\n"
                                utg_description += "• Some UI elements may not be visible - scroll if needed\n"

                    except Exception as e:
                        print(f"Error generating plan: {e}")





    utg_description += "--- End Navigation Guide ---\n\n"

    return utg_description


# Keep the original function for backward compatibility
def generate_utg_description(pure_utg_df, current_activity):
    """Original UTG description generator (for backward compatibility)."""
    return generate_utg_description_with_plan(pure_utg_df, current_activity, None, None)