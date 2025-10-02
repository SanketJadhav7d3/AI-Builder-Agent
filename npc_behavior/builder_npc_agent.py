
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools.base import StructuredTool
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from miney import Luanti, Node, Point
from typing import TypedDict, List, Any, Annotated
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode
import operator


class BuilderState(TypedDict):
    messages: Annotated[List[Any], operator.add]
    goal: str
    spawned: bool


class BuilderNPCAgent:

    def __init__(self, name, skin):
        self.name = name
        self.skin = skin
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        self.lt = Luanti(server="127.0.0.1", playername=self.name+"_brain", password="hello", port=30000)

        self.tools = [
            StructuredTool.from_function(self.get_player_pos),
            StructuredTool.from_function(self.get_nearby_blocks),
            StructuredTool.from_function(self.dig),
            StructuredTool.from_function(self.move_to),
            StructuredTool.from_function(self.place)
        ]
        
        # Bind tools to LLM
        self.agent_with_tools = self.llm.bind_tools(self.tools)
        # Create tool node for executing tools
        self.tool_node = ToolNode(self.tools)

    def get_player_pos(self):
        """Get player position as (x, y, z)."""
        try:
            pos = self.lt.players[self.name].position
            return f"Player position: ({pos.x}, {pos.y}, {pos.z})"
        except Exception as e:
            return f"Error getting position: {str(e)}"

    def get_nearby_blocks(self):
        """
        Scan the area around the player within the given radius and return a list of nearby blocks.
        Each entry includes the block type and its coordinates (x, y, z). Useful for understanding the environment,
        finding space to build, or locating specific resources.
        """
        try:
            # get player position
            player_pos = self.lt.players[self.name].position
            RADIUS = 10  # Reduced radius for better performance

            # store blocks
            blocks_info = []
            block_count = {}

            for x in range(player_pos.x-RADIUS, player_pos.x+RADIUS):
                for y in range(player_pos.y-RADIUS, player_pos.y+RADIUS):
                    for z in range(player_pos.z-RADIUS, player_pos.z+RADIUS):
                        point = Point(x, y, z)
                        node = self.lt.nodes.get(point)
                        if node and node.name != 'air':
                            block_type = node.name
                            blocks_info.append(f"{block_type} at ({x}, {y}, {z})")
                            block_count[block_type] = block_count.get(block_type, 0) + 1

            summary = f"Found {len(blocks_info)} blocks nearby. "
            summary += "Block types: " + ", ".join([f"{k}({v})" for k, v in block_count.items()])
            
            return summary + "\n" + "\n".join(blocks_info[:50])  # Limit output
        except Exception as e:
            return f"Error scanning blocks: {str(e)}"

    def dig(self, x: str, y: str, z: str):
        """Dig/remove the block at the given coordinates."""
        try:
            result = self.lt.lua.send_chat_command(f'aiv_dig {self.name} {x} {y} {z}')
            return f"Dug block at ({x}, {y}, {z})"
        except Exception as e:
            return f"Error digging at ({x}, {y}, {z}): {str(e)}"

    def move_to(self, x: str, y: str, z: str):
        """Move the player to a target position."""
        try:
            result = self.lt.lua.send_chat_command(f'aiv_move {self.name} {x} {y} {z}')
            return f"Moving to ({x}, {y}, {z})"
        except Exception as e:
            return f"Error moving to ({x}, {y}, {z}): {str(e)}"

    def place(self, block_type: str, x: str, y: str, z: str):
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

        Example usage:
            place('default:stone', 10, 5, 12)
        """
        try:
            result = self.lt.lua.send_chat_command(f'aiv_place {self.name} {block_type} {x} {y} {z}')
            return f"Placed {block_type} at ({x}, {y}, {z})"
        except Exception as e:
            return f"Error placing {block_type} at ({x}, {y}, {z}): {str(e)}"

    def spawn_player_direct(self):
        """Direct spawn without tool wrapper"""
        try:
            result = self.lt.lua.send_chat_command('aiv_spawn_one ' + self.name + ' ' + self.skin)
            return f"Player {self.name} spawned successfully {result}"
        except Exception as e:
            return f"Failed to spawn player: {str(e)}"

    def spawn_node(self, state: BuilderState):
        """Node to handle initial spawning"""
        if not state.get("spawned", False):
            spawn_result = self.spawn_player_direct()
            print('spawn_result', spawn_result)
            state["spawned"] = True
            state["messages"].append(AIMessage(content=f"{spawn_result}. Ready to work on goal: {state['goal']}"))

        return state

    def agent_node(self, state: BuilderState):
        """Main agent node that reasons about what to do next"""
        messages = state["messages"]
        
        # Add system context if this is the first agent call after spawning
        if len([m for m in messages if isinstance(m, HumanMessage)]) == 0:
            system_msg = HumanMessage(content=f"""
            Goal: {state['goal']}
            
            You are a builder agent in Minetest. Your goal is to accomplish the building task.
            
            Available tools:
            - get_player_pos: Get your current position
            - get_nearby_blocks: Scan the environment around you
            - move_to: Move to a specific location  
            - dig: Remove blocks at coordinates
            - place: Place blocks (use 'default:' prefix like 'default:wood', 'default:stone')
            
            Start by getting your position and scanning nearby blocks to understand your environment.
            Plan your approach step by step. When the goal is completed, end with "TASK_COMPLETED".
            """)

            messages = messages + [system_msg]
        
        print('messages', messages[-1].content, len(messages))
        # Get response from LLM
        response = self.agent_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(self, state: BuilderState):
        """Determine whether to continue, call tools, or end"""
        last_message = state["messages"][-1]
        
        # Check if task is completed
        if "TASK_COMPLETED" in last_message.content:
            return END
        
        # Check if LLM wants to call tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        
        # Otherwise continue with agent
        return "agent"

    def build_graph(self, goal: str):
        """Build the LangGraph with proper structure"""
        
        # Create workflow
        workflow = StateGraph(BuilderState)
        
        # Add nodes
        workflow.add_node("spawn", self.spawn_node)
        workflow.add_node("agent", self.agent_node)  
        workflow.add_node("tools", self.tool_node)
        
        # Add edges
        workflow.add_edge(START, "spawn")
        workflow.add_edge("spawn", "agent")
        workflow.add_edge("tools", "agent")  # After tools, go back to agent
        
        # Add conditional edges from agent
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                #"agent": "agent",    # Continue reasoning
                "tools": "tools",    # Execute tools
                END: END            # End the workflow
            }
        )
        
        # Compile with memory
        memory = MemorySaver()
        graph = workflow.compile(checkpointer=memory)
        
        return graph

    def run_build_task(self, goal: str, max_iterations: int = 50):
        """Convenience method to run a building task"""
        graph = self.build_graph(goal)
        
        initial_state = BuilderState(
            messages=[],
            goal=goal,
            spawned=False
        )
        
        # Run the graph with iteration limit
        config = {"configurable": {"thread_id": "builder_session"}, "recursion_limit": max_iterations}
        
        try:
            final_state = graph.invoke(initial_state, config)
            return final_state
        except Exception as e:
            print(f"Error during execution: {e}")
            return {"error": str(e), "messages": initial_state.get("messages", [])}


# Example usage:
if __name__ == "__main__":
    agent = BuilderAgent("builder_bot", "character_1")
    result = agent.run_build_task("Build a small 3x3 wooden house")
    
    print("Final result:")
    for msg in result.get("messages", []):
        print(f"{type(msg).__name__}: {msg.content}")

