
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain.tools.base import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from miney import Luanti, Node, Point
from typing import TypedDict, List, Any, Annotated, Dict, Tuple
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing import List, Tuple

class Tasks(BaseModel):
    overall_plan: str
    tasks: List[str]

MAX_MESSAGES = 100

def add_messages_trim(left, right):
    all_msgs = add_messages(left, right)
    return all_msgs[-MAX_MESSAGES:]


class BuilderState(TypedDict):
    messages: Annotated[List[Any], add_messages_trim]
    goal: str
    iteration_count: int
    last_tool_batch: List[str]
    pending_tasks: List[str]
    completed_tasks: List[str]


class PureToolNode:

    def __init__(self, tools: list):
        # Map tools by name for easy lookup
        self.tools = {t.name: t for t in tools}

    def __call__(self, state: BuilderState):
        tool_calls = state.get("last_tool_batch", [])
        if not tool_calls:
            return {}   # no update

        for call in tool_calls:
            name = call.get("name")
            args = call.get("args", {})
            tool = self.tools.get(name)

            if tool is None:
                print(f"[ToolNode] Unknown tool: {name}")
                continue

            try:
                result = tool.invoke(args)
                print(f"[ToolNode] {name} executed → {result}")
            except Exception as e:
                print(f"[ToolNode] {name} failed: {e}")

        return {}


class BuilderAgent:

    def __init__(self, name):
        self.name = name
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
        
        # Test connection to Minetest server
        try:
            self.lt = Luanti(server="127.0.0.1", playername=self.name, password="hello", port=30000)
            print(f"Successfully connected to Minetest as {self.name}")
        except Exception as e:
            print(f"Warning: Could not connect to Minetest server: {e}")
            self.lt = None

        self.tools = [
            StructuredTool.from_function(self.get_player_position),
            StructuredTool.from_function(self.place),
            StructuredTool.from_function(self.generate_absolute_floor),
            StructuredTool.from_function(self.generate_absolute_wall)
        ]
        
        # Bind tools to LLM
        self.agent_with_tools = self.llm.bind_tools(self.tools)
        # Create tool node for executing tools
        self.tool_node = PureToolNode(self.tools)

    def get_player_position(self):
        """
        Get the current position of the player in the Minetest world.

        This tool returns the player's exact coordinates (x, y, z) based on the
        server data. It is useful as the first step in any building or navigation
        task, since all subsequent block placements should be made relative to this
        position. The coordinates are integers representing the player's current 
        location in the world grid.

        Returns:
            str: A formatted string with the player's current position, for example:
                "Player position - x: 100, y: 5, z: 200"

        Raises:
            Exception: If the player cannot be found or the server connection fails.
        """
        position = self.lt.players[self.name].position

        return f'Player position - x: {position.x}, y: {position.y}, z: {position.z}'

    def get_nearby_blocks(self):
        """
        Scan the area around the player within the given radius and return a list of nearby blocks.
        Each entry includes the block type and its coordinates (x, y, z). Useful for understanding the environment,
        finding space to build, or locating specific resources.
        """
        try:
            # Check if player exists
            #if self.name not in self.lt.players:
                #return f"Player {self.name} not found in the game {list(self.lt.players)}"
            
            # Get player position
            player_pos = self.lt.players[self.name].position
            RADIUS = 5  # Reduced radius for better performance

            # Store blocks
            blocks_info = []
            block_count = {}

            player_x = int(player_pos.x)
            player_y = int(player_pos.y)
            player_z = int(player_pos.z)

            for x in range(player_x-RADIUS, player_x+RADIUS):
                for y in range(player_y, player_y+RADIUS):
                    for z in range(player_z-RADIUS, player_z+RADIUS):
                        point = Point(x, y, z)
                        node = self.lt.nodes.get(point)
                        if node and node.name != 'air':
                            block_type = node.name
                            blocks_info.append(f"{block_type} at ({int(x)}, {int(y)}, {int(z)})")
                            block_count[block_type] = block_count.get(block_type, 0) + 1

            if not blocks_info:
                return "No blocks found nearby (all air blocks)"

            summary = f"Found {len(blocks_info)} blocks nearby. "
            summary += "Block types: " + ", ".join([f"{k}({v})" for k, v in block_count.items()])
            
            return summary + "\n" + "\n".join(blocks_info[:50])  # Limit output
        except Exception as e:
            return f"Error scanning blocks: {str(e)}"

    def place(self, block_type: str, x: int, y: int, z: int):
        """
        Place a block of the given type at the specified coordinates.

        The block_type must always include the full node name with the 'default:' prefix,
        for example: 'default:wood', 'default:stone', 'default:dirt', etc.

        Common block types in Minetest (default game):
          - 'default:stone'
          - 'default:cobble'
          - 'default:wood'
          - 'default:tree'
          - 'default:leaves'
          - 'default:dirt'
          - 'default:grass'
          - 'default:sand'
          - 'default:glass'
          - 'default:brick'
          - 'default:fence_wood'
          - 'default:meselamp' for light source
          - 'air' to remove a block

        Example usage:
            place('default:stone', 10, 5, 12)
        """
        try:
            # Convert string coordinates to integers
            int_x, int_y, int_z = x, y, z
            
            # Create the point and node
            point = Point(int_x, int_y, int_z)

            node = Node(point.x, point.y, point.z, name=block_type)
            
            # Place the block
            self.lt.nodes.set(node)

            return f"Successfully placed {block_type} at ({x}, {y}, {z})"
            
        except ValueError as e:
            return f"Error: Invalid coordinates. x='{x}', y='{y}', z='{z}' must be integers: {str(e)}"
        except Exception as e:
            return f"Error placing {block_type} at ({x}, {y}, {z}): {str(e)}"

    def generate_absolute_floor(
        self,
        block_type: str,
        x0: int,
        y0: int,
        z0: int,
        width: int,
        length: int
    ) -> str:
        """
        Build a solid floor (1 block thick) at absolute coordinates.

        Args:
            block_type (str): Node name to place (e.g., "default:wood").
            x0 (int): Starting X coordinate.
            y0 (int): Y level (floor height).
            z0 (int): Starting Z coordinate.
            width (int): Width of the floor along the X axis.
            length (int): Length of the floor along the Z axis.

        Returns:
            str: Summary of placed floor with coordinates.
        """
        # Compute the far corner
        x1, y1, z1 = x0 + width - 1, y0, z0 + length - 1

        # Place all blocks in the floor
        for x in range(x0, x0 + width):
            for z in range(z0, z0 + length):
                self.place(block_type=block_type, x=x, y=y0, z=z)

        return f"Placed {block_type} floor from ({x0}, {y0}, {z0}) to ({x1}, {y1}, {z1})"

    def generate_absolute_wall(
        self,
        block_type: str,
        x0: int,
        y0: int,
        z0: int,
        height: int,
        length: int,
        axis: str = "x",
    ) -> str:
        """
        Build a vertical wall at absolute coordinates.

        Args:
            block_type (str): e.g. 'default:wood'
            x0 (int): Starting X coordinate.
            y0 (int): Starting Y coordinate (base of wall).
            z0 (int): Starting Z coordinate.
            height (int): Wall height along Y.
            length (int): Wall length along chosen axis.
            axis (str): 'x' -> wall extends along X,
                        'z' -> wall extends along Z.
            hollow (bool): Ignored (walls are always 1 block thick).

        Returns:
            str: Summary of placed wall with coordinates.
        """
        # Compute end coordinates
        if axis == "x":
            x1, y1, z1 = x0 + length - 1, y0 + height - 1, z0
        elif axis == "z":
            x1, y1, z1 = x0, y0 + height - 1, z0 + length - 1
        else:
            raise ValueError(f"Invalid axis '{axis}', must be 'x' or 'z'")

        # Place the wall
        for dy in range(height):
            for d in range(length):
                x = x0 + d if axis == "x" else x0
                z = z0 + d if axis == "z" else z0
                y = y0 + dy
                self.place(block_type=block_type, x=x, y=y, z=z)

        return f"Placed {block_type} wall from ({x0}, {y0}, {z0}) to ({x1}, {y1}, {z1})"

    def planning_node(self, state: BuilderState):

        structured_llm = self.llm.with_structured_output(Tasks)

        prompt = f"""
            You are a planning agent for a Minetest builder.

            Your job is to take the given building goal and translate it into a structured construction plan.  
            You must output two fields:

            1. overall_plan — a detailed multi-sentence (or even multi-paragraph) explanation that fully summarizes the structure, materials, layout, and intended build order.  
            There is no limit to how long this description can be. Make it as detailed as necessary.  

            2. tasks — a sequential list of substructure tasks that break the goal into concrete build steps.  
            There is no upper limit to the number of tasks. Include as many tasks as required to fully capture the build.  
            Each task must clearly state:  
            - The material(s) used (e.g., default:stone, default:wood, default:glass),  
            - The type of substructure (floor, wall, roof, staircase, door, window, lighting, decoration, garden, furniture, etc.),  
            - The key coordinate range(s), written as `(x1,y1,z1) to (x2,y2,z2)`.  

            Guidelines:
            - Tasks must be ordered logically so they can be executed step by step (e.g., base before walls, walls before roof, interior before final decorations).  
            - Keep tasks at the **substructure level** — never expand into individual block placements.  
            - If the goal includes decorations, lighting, furniture, or garden elements, include them as separate tasks at the appropriate stage.  
            - Do not invent new features beyond what is described in the goal.  

            Examples:

            overall_plan: "The house will be a compact 3x3 stone structure with a wooden roof. 
            I will first build a stone floor, then raise the walls, add windows and a door, place the roof, and finally decorate with lighting and furniture."

            tasks:
            - "Build a 3x3 stone floor from (0,0,0) to (2,0,2)."  
            - "Construct a 3-block high stone wall from (0,1,0) to (2,3,0)."  
            - "Add a wooden roof from (0,4,0) to (2,4,2)."  
            - "Install a wooden door (doors:door_wood) at (1,1,0)."  
            - "Place glass windows (default:glass) in the wall section from (0,2,1) to (0,3,1)."  
            - "Add lighting using meselamps (default:meselamp) at ceiling coordinates (1,3,1)."  
            - "Decorate the room with a bed (beds:bed) at (1,1,1) and a chest (default:chest) at (2,1,1)."  
            - "Place torches (default:torch) on the walls from (0,2,0) to (2,2,0)."  
            - "Create a small garden with dirt and flowers (default:dirt_with_grass + flowers:rose) from (3,0,0) to (5,0,2)."  


            The goals is as follows.

            {state['goal']}
        """

        tasks_obj: Tasks = structured_llm.invoke(prompt)

        # Print reasoning text so user can see it
        print("=== PLANNING / INTERNAL REASONING ===")

        print(tasks_obj.overall_plan)
        print("\nPending tasks")
        for i, task in enumerate(tasks_obj.tasks):
            print(i, task)

        print("====================================")

        return {
            "messages": state["messages"] + [
                AIMessage(content=f"Overall Plan: {tasks_obj.overall_plan}")
            ],
            "pending_tasks": tasks_obj.tasks
        }

    def agent_node(self, state: BuilderState):

        if not state.get("pending_tasks"):
            return {"messages": state["messages"] + [AIMessage(content="No tasks left. TASK_COMPLETED")]}

        current_task = state["pending_tasks"][0]
        completed_tasks = state.get("completed_tasks", [])

        system_prompt = (
            "You are a builder agent operating inside a Minetest world.\n\n"
            "You receive one high-level building task at a time, such as:\n"
            "  'Build a 3x3 stone floor from (0,0,0) to (2,0,2)'.\n\n"
            "Your job is to translate this task into concrete tool calls using the available APIs:\n"
            "  - place(block_type, x, y, z)\n"
            "  - generate_absolute_floor(block_type, x0, y0, z0, width, length)\n"
            "  - generate_absolute_wall(block_type, x0, y0, z0, height, length, axis)\n\n"
            "Guidelines:\n"
            "1. Always use the task description as the single source of truth. Do not invent new plans.\n"
            "2. Convert coordinate ranges (x1,y1,z1 to x2,y2,z2) into correct width/height/length values.\n"
            "3. Place entire substructures in one batch call (do not split a single wall or floor across steps).\n"
            "4. If a task can be done with a higher-level tool (like floor or wall), prefer that instead of many place calls.\n"
            "5. Return both natural language context (what you’re building) and the tool calls.\n"
        )

        human_prompt = f"""
        Current task:
        {current_task}

        Completed tasks so far:
        {completed_tasks if completed_tasks else "None"}
        """
        print("="*10 + "AGENT NODE HUMAN_PROMPT" + "="*10)
        print(human_prompt)
        print("="*25)

        conversation = [SystemMessage(content=system_prompt)]
        conversation.extend(state["messages"])  # include prior plan + summaries
        conversation.append(HumanMessage(content=human_prompt))

        response = self.agent_with_tools.invoke(conversation)

        tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
        execution_text = response.content or f"Executing task: {current_task}"


        return {
            "messages": state["messages"] + [AIMessage(content=execution_text)],
            "last_tool_batch": tool_calls,
            "pending_tasks": state["pending_tasks"][1:],  # move to next
            "completed_tasks": state.get("completed_tasks", []) + [current_task],
        }

    def summarize_tools_node(self, state: BuilderState):

        last_batch = state.get('last_tool_batch', [])
        
        if not last_batch:
            return {}

        planning_msg = None

        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                planning_msg = msg.content
                break

        if not planning_msg:
            planning_msg = "(No planning message found)"

        sys_msg = SystemMessage(content=(
            "You are an assistant that summarizes the outcome of executed tool calls. "
            "Given the agent's plan and the executed tools, produce a short natural language "
            "description of what has been built, highlighting key coordinates, walls, floors, doors, etc. "
            "Also mention what tools were called."
            "Output should be as detailed as possible with crucial information such as player coordinates and essential block coordinates"
        ))

        human_msg = HumanMessage(content=(
            f"Agent plan: {planning_msg}\n"
            f"Executed tools (JSON): {last_batch}\n\n"
            "Write a detailed natural language summary of what was built with important coordinates."
        ))

        resp = self.llm.invoke([sys_msg, human_msg])

        summary_text = f"SUMMARY OF TOOL CALLS:\n {resp.content}" or "Built some structures (details in executed tools)."
 
        print("="*10 + "SUMMARY NODE" + "="*10)

        print(summary_text)

        print("="*25)

        return {
            "messages": state["messages"] + [AIMessage(content=summary_text)]
        }

    def should_continue(self, state: BuilderState):
        """Determine whether to continue, call tools, or end"""

        if not state.get('messages'):
            return END

        last_message = state['messages'][-1]

        content = getattr(last_message, "content", "")
        if content and "TASK_COMPLETED" in str(content):
            print("[should_continue] Task completed detected in last message.")
            return END

        if state.get("iteration_count", 0) >= 1000:
            print("[should_continue] Max iterations reached!")
            return END

        last_batch = state.get("last_tool_batch", [])

        if last_batch:
            print(f"[should_continue] {len(last_batch)} tool calls found. Proceeding to tools node.")
            return "tools"

        print("[should_continue] No tool calls requested and task not completed. Ending workflow.")
        return END


    def increment_counter(self, state: BuilderState):
        """Helper node to increment iteration counter"""
        return {"iteration_count": state.get("iteration_count", 0) + 1}

    def build_graph(self, goal: str):
        """Build the LangGraph with proper structure"""
        
        # Create workflow
        workflow = StateGraph(BuilderState)
        
        # Add nodes
        workflow.add_node("planning", self.planning_node)  
        workflow.add_node("agent", self.agent_node)  
        workflow.add_node("tools", self.tool_node)
        workflow.add_node("increment", self.increment_counter)
        workflow.add_node("summarizer", self.summarize_tools_node)
        
        # Add edges
        workflow.add_edge(START, "planning")
        workflow.add_edge("planning", "increment")
        workflow.add_edge("increment", "agent")
        workflow.add_edge("tools", "summarizer")  # After tools, summarize the tool call information
        workflow.add_edge("summarizer", "increment")  # After tools, increment counter then go back to agent
        
        # Add conditional edges from agent
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "tools": "tools",    # Execute tools
                END: END            # End the workflow
            }
        )
        
        # Compile with memory
        memory = MemorySaver()
        graph = workflow.compile(checkpointer=memory)

        try:
            png_data = graph.get_graph().draw_mermaid_png()

            with open("graph_output.png", "wb") as f:
                f.write(png_data)

            print("Graph saved as graph_output.png")

        except Exception:
            # This requires some extra dependencies and is optional
            pass
        
        return graph

    def run_build_task(self, goal: str, max_iterations: int = 1000):
        """Convenience method to run a building task"""
        try:
            # Test LLM connection first
            print("Testing LLM connection...")
            test_response = self.llm.invoke([HumanMessage(content="Hello, are you working?")])
            print(f"LLM test successful: {test_response.content[:50]}...")
        except Exception as e:
            print(f"LLM test failed: {e}")
            return {"error": f"LLM connection failed: {str(e)}"}
        
        graph = self.build_graph(goal)
        
        initial_state = BuilderState(
            messages=[],
            goal=goal,
            iteration_count=0,
            last_tool_batch=[]
        )
        
        # Run the graph with iteration limit
        config = {
            "configurable": {"thread_id": "builder_session"}, 
            "recursion_limit": max_iterations + 10  # Add buffer for internal nodes
        }
        
        try:
            print(f"Starting build task: {goal}")
            final_state = graph.invoke(initial_state, config)
            print("Build task completed!")
            return final_state
        except Exception as e:
            print(f"Error during execution: {e}")
            # Print more detailed error info
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return {"error": str(e), "messages": initial_state.get("messages", [])}


# Example usage:
if __name__ == "__main__":
    # Create builder agent
    builder = BuilderAgent("TestBot")
    
    # Run a building task
    result = builder.run_build_task("Build a small 3x3 stone house")
    
    # Print results
    if "error" in result:
        print(f"Task failed: {result['error']}")
    else:
        print("Task completed successfully!")
        print(f"Total messages: {len(result.get('messages', []))}")
