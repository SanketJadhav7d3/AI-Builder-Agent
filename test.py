
from miney import Luanti
import threading
import time
from npc_behavior import npc
from npc_behavior.builder_agent import BuilderAgent


# Example usage:
if __name__ == "__main__":
    # Create builder agent
    builder = BuilderAgent("testbot")

    goal = """
    Build a grand Floating House complex at absolute coordinates x=50, y=50, z=0, fully suspended in the air with no ground connection.  
The entire floating platform footprint should measure exactly 100×100 blocks, with the house positioned centrally and a surrounding garden filling the rest of the space.  

=== Structure & Materials (basic blocks only) ===
- Walls: wood (default:wood), stone (default:stone), and windows with glass (default:glass).  
- Floors: wooden planks (default:wood).  
- Roof: flat or gently sloped, made from stone (default:stone) and wood (default:wood), with battlements or fence edging.  
- Doors: wooden doors (doors:door_wood) at entrances and between rooms.  
- Windows: glass blocks (default:glass) placed symmetrically for balance and light.  
- Lighting: meselamps (default:meselamp) embedded in walls/ceilings, torches (default:torch) for atmosphere.  
- Garden: dirt_with_grass (default:dirt_with_grass), fences (default:fence_wood), flowers (flowers:*), bushes/trees (default:leaves, default:tree), and benches made from fences (default:fence).  

=== Layout & Features ===
- **Floating Base Platform (100×100, y=50):**  
  - Entire footprint made from stone and wood mix, edged with fences for safety.  
  - Outer ring designed as a large decorative floating garden with flowers, meselamps, and benches.  
  - Central area reserved for the multi-story house.  

- **Main House (centered, ~40×40 footprint):**  
  - Multi-story structure (3–4 floors).  
  - **Ground Floor (y=50):** entrance hall, large common area, storage rooms.  
  - **Second Floor (y≈60):** bedrooms, private chambers, and hallways.  
  - **Third Floor (y≈70):** study/library, decorative balcony areas with windows.  
  - **Fourth Floor (y≈80):** rooftop terrace with battlements, seating, and garden planters.  

- **Garden Spaces (within outer ring of the 100×100 base):**  
  - Dirt_with_grass foundation with colorful flowers (flowers:rose, flowers:tulip, flowers:dandelion_white, flowers:dandelion_yellow).  
  - Trees made with default:tree and default:leaves, scattered for variety.  
  - Decorative meselamps at corners and fence posts, with torches along the paths.  
  - Benches made from fences (default:fence) and optional chests for storage.  

=== Connectivity ===
- Wooden or stone staircases connecting floors inside the house.  
- Open ceiling clearance above staircases.  
- Doors connecting rooms, hallways, and leading outside into the garden.  
- Pathways from the central house to the garden areas.  

=== Lighting & Decoration ===
- Meselamps embedded in walls, ceilings, and garden paths for consistent lighting.  
- Torches placed in corners, corridors, balconies, and garden posts for atmosphere.  
- Beds, chests, and tables (fences) indoors for livability.  
- Symmetrical window placement for aesthetics and balanced views.  
- Flowers, trees, and bushes outdoors for a vibrant, lively environment.  

Final design should feel like a **majestic floating villa complex**: a vast 100×100 platform suspended in the sky at (50,50,0), with a central multi-story home surrounded by gardens, full of light, decoration, and functionality.  

    """
    
    # Run a building task
    result = builder.run_build_task(goal)
    
    # Print results
    if "error" in result:
        print(f"Task failed: {result['error']}")
    else:
        print("Task completed successfully!")
        print(f"Total messages: {len(result.get('messages', []))}")

