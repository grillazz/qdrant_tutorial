import os

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

from transformers import AutoTokenizer
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv

load_dotenv()

collection = 'jira_task_search_6'

encoder = SentenceTransformer("all-MiniLM-L6-v2")

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# For ANN/HNSW:
# client = QdrantClient(url="http://localhost:6333")

# Create collection with three named vectors
if not client.collection_exists(collection_name=collection):
    client.create_collection(
        collection_name=collection,
        vectors_config={
            'fixed': models.VectorParams(size=384, distance=models.Distance.COSINE),
            'sentence': models.VectorParams(size=384, distance=models.Distance.COSINE),
            'semantic': models.VectorParams(size=384, distance=models.Distance.COSINE),
        },
    )

# Create index for the 'chunking' field to allow filtering (idempotent if handled by status check or just try-except)
collection_info = client.get_collection(collection_name=collection)
if "chunking" not in collection_info.payload_schema:
    client.create_payload_index(
        collection_name=collection,
        field_name="chunking",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

# Step 3: Implementing the Chunking Strategies
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
MAX_TOKENS = 100
# 256 bad ide as 1st and 2bd are not good match

def fixed_size_chunks(text, size=MAX_TOKENS):
    """Fixed-size chunking: splits at exact token boundaries"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return [
        tokenizer.decode(tokens[i:i + size], skip_special_tokens=True)
        for i in range(0, len(tokens), size)
    ]


def sentence_chunks(text):
    """Sentence-aware chunking: respects sentence boundaries"""
    splitter = SentenceSplitter(chunk_size=MAX_TOKENS, chunk_overlap=20)
    return splitter.split_text(text)


def semantic_chunks(text):
    """Semantic chunking: uses embedding similarity to find natural breaks.
    Note: still constrained by the embed model's context window (same as retrievers)."""
    from llama_index.core import Document

    semantic_splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    nodes = semantic_splitter.get_nodes_from_documents([Document(text=text)])
    return [node.text for node in nodes]


# Step 4: Processing and Uploading the Data
points = []
idx = 0

jira_data = documents = [
  {
    "issue_key": "GARDEN-001",
    "summary": "Prepare tomato seed trays",
    "description": "Prepare a set of seed trays by filling each cell with a high‑quality sterile starter mix that retains moisture but still drains well. Gently press the mix into the trays so it is level but not compacted, then sow tomato seeds at a shallow depth of about 0.5 cm, following the recommended spacing per variety. Use a permanent marker or labels to clearly identify each tray and variety so plants can be tracked from germination through transplanting. Place the trays in a designated warm propagation area with consistent temperature and indirect light, and monitor daily for moisture levels to avoid overwatering. Keep the trays in this environment until seedlings develop their first true leaves, at which point they can be hardened off and moved to the next growth stage. This step is critical for ensuring healthy tomato plants, strong root development, and good yields later in the season, while also minimizing the risk of diseases such as damping‑off that can affect weak seedlings.",
    "labels": ["gardening", "vegetables", "seed-starting", "tomato"],
    "priority": "Medium",
    "status": "To Do"
  },
  {
    "issue_key": "GARDEN-002",
    "summary": "Inspect drip irrigation in strawberry bed",
    "description": "Inspect the entire drip line layout in the strawberry bed for leaks, clogged emitters, and pressure issues that could affect uniform watering. Walk the rows slowly and check each emitter for proper flow, replacing any damaged or blocked fittings immediately. Verify that the main line is securely connected to the water source and that the pressure regulator is functioning to prevent over‑pressure in the system. Adjust the timer settings if needed to match current soil moisture and weather conditions. Record any problem areas and plan follow‑up checks to ensure the bed remains well‑irrigated throughout the fruiting season. Proper irrigation management is essential to maintain strong strawberry plants, prevent water stress, and reduce the risk of fungal diseases that thrive in poorly watered or over‑watered conditions.",
    "labels": ["gardening", "fruit", "irrigation", "maintenance", "strawberry"],
    "priority": "High",
    "status": "In Progress"
  },
  {
    "issue_key": "GARDEN-003",
    "summary": "Weed carrot rows before mulching",
    "description": "Carefully remove all visible weeds from the carrot rows by hand or using a small hoe, taking care not to disturb the developing carrot roots near the surface. Focus on removing both broad‑leaf and grass‑type weeds that compete for nutrients, water, and light. Once the bed is mostly clear of weeds, apply a thin, even layer of organic mulch such as straw or compost over the soil to suppress future weed germination and reduce evaporation. Avoid piling mulch directly against young carrot stems to prevent rot or pest issues. This maintenance step helps ensure that carrots grow with minimal competition, improves soil moisture retention, and reduces the need for frequent weeding later in the season.",
    "labels": ["gardening", "vegetables", "weeding", "mulching", "carrot"],
    "priority": "Medium",
    "status": "To Do"
  },
  {
    "issue_key": "GARDEN-004",
    "summary": "Prune apple tree water sprouts",
    "description": "Inspect the canopy of the apple tree for vertical water sprouts that grow straight upward from larger branches or the trunk, as these often steal energy from fruiting wood. Use clean, sharp pruning shears to remove these sprouts at their base, leaving the main branching structure intact. Focus on areas where airflow and light penetration appear restricted, and avoid over‑pruning in a single season to prevent stress. After pruning, rake away any fallen debris to reduce the risk of disease and pests harboring in the leaf litter. Proper pruning improves air circulation through the canopy, encourages outward‑facing fruiting branches, and helps maintain a manageable tree height while supporting higher fruit quality and yield over time.",
    "labels": ["gardening", "fruit", "pruning", "tree-care", "apple"],
    "priority": "High",
    "status": "To Do"
  },
  {
    "issue_key": "GARDEN-005",
    "summary": "Start cucumber transplants in greenhouse",
    "description": "Fill individual pots or trays with a well‑draining seed‑starting mix and sow cucumber seeds at a shallow depth, following the recommended spacing per type. Place the containers in a warm, well‑lit area inside the greenhouse, ensuring that the temperature remains stable and the seedlings are not exposed to drafts or cold spots. Keep the soil consistently moist but not saturated, and monitor daily for signs of germination or disease. Once seedlings develop their first true leaves, begin to harden them off by gradually increasing exposure to outside conditions before transplanting. Starting cucumbers in the greenhouse gives them an early growth advantage, reduces the risk of soil‑borne pests, and provides a controlled environment where temperature and moisture can be closely managed.",
    "labels": ["gardening", "vegetables", "greenhouse", "seed-starting", "cucumber"],
    "priority": "Medium",
    "status": "In Progress"
  },
  {
    "issue_key": "GARDEN-006",
    "summary": "Harvest ripe raspberries",
    "description": "Pick fully ripe raspberries in the morning when temperatures are cooler to minimize stress on the plants and preserve fruit quality. Use gentle rolling motions with your fingers so that berries detach easily without crushing nearby fruit. Place harvested berries into shallow containers lined with paper or soft cloth to prevent bruising, and remove any visibly damaged or overripe fruit before storage. Move the containers quickly to a cool area or refrigeration unit to slow deterioration and extend shelf life. Regular harvesting helps prevent over‑ripening on the plant, encourages continued fruiting, and reduces the risk of mold or pest infestations that thrive on dropped or rotting berries.",
    "labels": ["gardening", "fruit", "harvest", "raspberry"],
    "priority": "High",
    "status": "To Do"
  },
  {
    "issue_key": "GARDEN-007",
    "summary": "Amend raised bed with compost",
    "description": "Spread a layer of mature, well‑rotted compost evenly over the surface of the raised vegetable bed, aiming for a depth that is neither too shallow nor so thick that it compacts the soil. Use a fork or rake to lightly incorporate the compost into the top few inches of soil, taking care not to disturb deeper root structures if plants are already established. Level the surface after incorporation so that future planting or seeding can be done evenly. This step improves soil structure, increases organic matter content, and provides a slow‑release source of nutrients for vegetables over the growing season. Regular compost application also helps maintain soil life and supports healthier plants with better resistance to pests and diseases.",
    "labels": ["gardening", "soil", "compost", "vegetables", "bed-prep"],
    "priority": "Medium",
    "status": "Done"
  },
  {
    "issue_key": "GARDEN-008",
    "summary": "Monitor aphids on bean plants",
    "description": "Inspect the undersides of bean leaves and growing tips for clusters of small, soft‑bodied aphids that often form dense colonies. Use a hand lens if necessary to distinguish aphids from other insects or debris, and record the severity of infestation on a simple scale. If aphid numbers exceed the threshold for acceptable damage, apply an approved organic control such as insecticidal soap or neem oil, taking care to cover affected areas without over‑spraying nearby beneficial insects. Monitor again after treatment to ensure the population is declining. Regular inspection helps catch outbreaks early, prevents yield loss, and supports a more sustainable approach by avoiding unnecessary broad‑spectrum pesticide use.",
    "labels": ["gardening", "vegetables", "pest-control", "beans", "inspection"],
    "priority": "High",
    "status": "In Progress"
  },
  {
    "issue_key": "GARDEN-009",
    "summary": "Thin beet seedlings",
    "description": "Carefully remove excess beet seedlings from overcrowded areas so that remaining plants are spaced according to recommended guidelines, typically several centimeters apart. Use small scissors or fingers to snip out weaker or smaller seedlings, leaving the strongest ones to develop into mature roots. Avoid pulling seedlings by the base, as this can disturb the roots of neighboring plants. Water the bed lightly after thinning to reduce stress on the surviving plants and help them recover. Proper thinning ensures that each beet has enough space, light, and nutrients to form a well‑shaped root without competing too heavily with its neighbors.",
    "labels": ["gardening", "vegetables", "thinning", "beet"],
    "priority": "Medium",
    "status": "To Do"
  },
  {
    "issue_key": "GARDEN-010",
    "summary": "Check soil moisture in peach orchard",
    "description": "Use a soil moisture probe or sensor to measure moisture levels in the root zone around peach trees, focusing on representative sites within the orchard. Compare readings with historical data and current weather conditions to determine whether irrigation is needed or should be reduced. If soil is drying too quickly, adjust the irrigation schedule or flow rate to maintain consistent moisture without waterlogging. Record any areas that consistently dry out or remain too wet, as these may indicate uneven distribution or drainage issues. Maintaining proper soil moisture is crucial for fruit set, fruit size, and overall tree health, especially during key growth stages like flowering and fruit expansion.",
    "labels": ["gardening", "fruit", "orchard", "soil-moisture", "peach"],
    "priority": "High",
    "status": "In Progress"
  },
  {
    "issue_key": "GARDEN-011",
    "summary": "Install trellis for pole beans",
    "description": "Set up a sturdy trellis system along the pole bean row, using posts driven into the ground at regular intervals and horizontal supports or netting to provide climbing structure. Secure the trellis so that it can withstand wind, rain, and the weight of mature plants and fruit. Guide young bean vines toward the trellis as they grow, using twine or plant clips if necessary, and check regularly for any loosening of supports. A well‑installed trellis keeps plants off the ground, improves airflow around foliage, reduces disease risk, and makes harvesting easier by keeping beans within reach. This setup also maximizes vertical space and can increase yield by allowing more plants in a smaller footprint.",
    "labels": ["gardening", "vegetables", "support", "trellis", "beans"],
    "priority": "Medium",
    "status": "To Do"
  },
  {
    "issue_key": "GARDEN-012",
    "summary": "Remove mildew from squash leaves",
    "description": "Inspect squash plants for signs of powdery mildew, which appears as white, powdery patches on leaves and stems. Use clean, sterilized pruning shears to remove heavily infected leaves and dispose of them away from the garden to prevent spore spread. If the infection is mild, consider applying an approved fungicide or organic treatment according to label instructions, being careful not to over‑treat healthy foliage. Monitor neighboring plants closely, as mildew can spread rapidly under warm, humid conditions. Managing mildew early helps preserve photosynthetic leaf area, supports fruit development, and reduces the chance of secondary pests taking advantage of weakened plants.",
    "labels": ["gardening", "vegetables", "disease-control", "squash", "inspection"],
    "priority": "High",
    "status": "In Progress"
  },
  {
    "issue_key": "GARDEN-013",
    "summary": "Pick ripe strawberries for market",
    "description": "Harvest ripe strawberries in the early morning when fruits are cool and firm, carefully twisting each berry off the stem to avoid damaging the plant. Place berries into shallow containers that are lined with soft material to prevent bruising, and sort out any overripe, damaged, or moldy fruit before packing. Arrange the containers in a shaded area or refrigerator until ready for transport, and aim to keep them moving quickly from field to market or storage. Proper harvesting techniques help maintain fruit quality, reduce post‑harvest losses, and support repeat sales by delivering consistently fresh, attractive strawberries to customers.",
    "labels": ["gardening", "fruit", "harvest", "strawberry", "market"],
    "priority": "High",
    "status": "To Do"
  },
  {
    "issue_key": "GARDEN-014",
    "summary": "Turn compost pile",
    "description": "Use a pitchfork or compost turner to mix the outer layers of the compost pile into the center, where decomposition is most active. Work systematically from one end to the other, breaking up any compacted clumps and checking for moisture by feel; if the pile is too dry, lightly sprinkle water while turning. If it is too wet, add drier brown material such as straw or shredded cardboard. Turning the pile introduces oxygen, speeds up microbial activity, and helps achieve a more uniform, finished compost. Regular turning improves the final product’s texture and nutrient balance, making it better suited for amending garden soil later in the season.",
    "labels": ["gardening", "compost", "maintenance", "soil"],
    "priority": "Medium",
    "status": "Done"
  },
  {
    "issue_key": "GARDEN-015",
    "summary": "Plant lettuce succession crop",
    "description": "Prepare a section of the cool bed by raking the surface to remove debris and ensure a fine, even seedbed. Sow lettuce seeds in rows or blocks at the recommended spacing, lightly rake or press them into the soil, and water gently to avoid washing seeds away. Mark the area with a label indicating the variety and sowing date so that future maintenance and harvesting can be tracked. Planting a succession crop helps maintain a continuous supply of fresh lettuce through the season, filling gaps left by earlier harvests and making efficient use of available space. Keeping the soil cool and well‑moistened during germination supports strong, uniform stands and reduces the risk of bolting in hot weather.",
    "labels": ["gardening", "vegetables", "planting", "succession", "lettuce"],
    "priority": "Medium",
    "status": "To Do"
  },
  {
    "issue_key": "GARDEN-016",
    "summary": "Inspect plum tree for insect damage",
    "description": "Carefully examine plum tree leaves, twigs, and fruit for signs of insect feeding, such as holes, notches, discoloration, or frass. Look especially along the edges of leaves and around developing fruit, where pests like caterpillars or beetles often concentrate. If damage exceeds a defined threshold, note affected branches and decide whether to apply targeted control measures or rely on natural predators. Record findings in a log so trends over time can inform future prevention strategies. Regular inspection helps catch pest problems early, protect fruit quality, and reduce the need for broad‑spectrum treatments that can harm beneficial insects.",
    "labels": ["gardening", "fruit", "inspection", "pest-control", "plum"],
    "priority": "High",
    "status": "To Do"
  },
  {
    "issue_key": "GARDEN-017",
    "summary": "Water newly planted pepper seedlings",
    "description": "Water pepper seedlings at the base using a gentle flow or drip method, ensuring that the root zone is moist without saturating the soil. Avoid overhead watering that wets the foliage excessively, as this can encourage fungal diseases. Check the soil moisture daily or every other day, depending on weather and soil type, and adjust the amount or frequency as needed. As the plants grow, gradually increase the volume of water to match their expanding root system. Adequate moisture during this early stage supports strong root establishment, reduces transplant shock, and helps peppers develop sturdy stems and healthy leaves before flowering begins.",
    "labels": ["gardening", "vegetables", "watering", "seedlings", "pepper"],
    "priority": "Medium",
    "status": "In Progress"
  },
  {
    "issue_key": "GARDEN-018",
    "summary": "Harvest zucchini before overgrowth",
    "description": "Pick zucchini when the fruits are still small to medium‑sized and their skins are tender, cutting them cleanly with a sharp knife or shears near the stem. Avoid leaving large, overgrown zucchinis on the plant, as they can sap energy that would otherwise support new flowers and smaller fruits. Inspect the plants regularly, especially during warm periods when growth is rapid, and remove any damaged or diseased fruit as you harvest. Frequent harvesting encourages the plant to continue producing and can extend the productive life of the crop. Smaller zucchinis are also generally better for cooking and storage, with firmer texture and fewer seeds.",
    "labels": ["gardening", "vegetables", "harvest", "zucchini"],
    "priority": "High",
    "status": "To Do"
  },
  {
    "issue_key": "GARDEN-019",
    "summary": "Replace damaged mulch around blueberry bushes",
    "description": "Remove any compacted, matted, or decomposed mulch from around blueberry bushes, taking care not to disturb the shallow root system during the process. Spread a fresh layer of suitable organic mulch such as pine bark or wood chips, leaving a small gap around the trunk to avoid moisture buildup against the stem. Keep the mulch layer at a moderate depth so that it conserves moisture and suppresses weeds without restricting gas exchange at the soil surface. Refreshing the mulch helps maintain consistent soil moisture, moderates temperature extremes, and slowly adds organic matter back into the soil as it breaks down.",
    "labels": ["gardening", "fruit", "mulching", "blueberry", "maintenance"],
    "priority": "Medium",
    "status": "Done"
  }
]

for jira_task in jira_data:  # Process each jira_task
    # Fixed-size chunks
    for chunk in fixed_size_chunks(jira_task["description"]):
        points.append(models.PointStruct(
            id=idx,
            vector={"fixed": encoder.encode(chunk).tolist()},
            payload={**jira_task, "chunk": chunk, "chunking": "fixed"}
        ))
        idx += 1

    # Sentence-aware chunks
    for chunk in sentence_chunks(jira_task["description"]):
        points.append(models.PointStruct(
            id=idx,
            vector={"sentence": encoder.encode(chunk).tolist()},
            payload={**jira_task, "chunk": chunk, "chunking": "sentence"}
        ))
        idx += 1

    # Semantic chunks
    for chunk in semantic_chunks(jira_task["description"]):
        points.append(models.PointStruct(
            id=idx,
            vector={"semantic": encoder.encode(chunk).tolist()},
            payload={**jira_task, "chunk": chunk, "chunking": "semantic"}
        ))
        idx += 1

client.upsert(collection_name=collection, points=points)
print(f"Uploaded {idx} vectors across three chunking strategies")


# Step 5: Comparing Search Results
def search_and_compare(query, k=3):
    """Compare search results across all three chunking strategies"""
    print(f"Query: '{query}'\n")

    for strategy in ['fixed', 'sentence', 'semantic']:
        results = client.query_points(
            collection_name=collection,
            query=encoder.encode(query).tolist(),
            using=strategy,
            limit=k,
        )

        print(f"--- {strategy.upper()} CHUNKING ---")
        for i, point in enumerate(results.points, 1):
            payload = point.payload
            print(f"{i}. {payload['issue_key']} ({payload['summary']}) | Score: {point.score:.3f}")
            print(f"   Chunk: {payload['chunk'][:100]}...")
        print()


def analyze_chunking_effectiveness():
    """Analyze which chunking strategy works best for your domain"""

    print("CHUNKING STRATEGY ANALYSIS")
    print("=" * 40)

    # Get chunk statistics for each strategy
    for strategy in ["fixed", "sentence", "semantic"]:
        # Count chunks per strategy
        results = client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="chunking", match=models.MatchValue(value=strategy)
                    )
                ]
            ),
            limit=100,
        )

        chunks = results[0]
        if not chunks:
            print(f"\n{strategy.upper()} STRATEGY: No chunks found")
            continue

        chunk_sizes = [len(chunk.payload["chunk"]) for chunk in chunks]

        print(f"\n{strategy.upper()} STRATEGY:")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Avg chunk size: {sum(chunk_sizes)/len(chunk_sizes):.0f} chars")
        print(f"  Size range: {min(chunk_sizes)}-{max(chunk_sizes)} chars")





# Test with different queries
search_and_compare("prevent water stress, and reduce the risk of fungal diseases that thrive in poorly watered or over‑watered conditions. ")


analyze_chunking_effectiveness()
