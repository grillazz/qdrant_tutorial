from nltk.corpus import movie_reviews
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

from transformers import AutoTokenizer
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Step 2: The Three-Vector Experiment
# Initialize components
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# In-memory for demo: NO HNSW built -> queries are a full scan.
client = QdrantClient(":memory:")

# For ANN/HNSW:
# client = QdrantClient(url="http://localhost:6333")

# Create collection with three named vectors
client.create_collection(
    collection_name='movie_search',
    vectors_config={
        'fixed': models.VectorParams(size=384, distance=models.Distance.COSINE),
        'sentence': models.VectorParams(size=384, distance=models.Distance.COSINE),
        'semantic': models.VectorParams(size=384, distance=models.Distance.COSINE),
    },
)

# Step 3: Implementing the Chunking Strategies
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
MAX_TOKENS = 40


# MAX_TOKENS = 40 is set because it is the chunk size, not the embedding model’s maximum context length. The notebook is splitting long movie descriptions into small pieces so each piece can be embedded safely, even though all-MiniLM-L6-v2 can handle up to 256 tokens.
#
# So the logic is:
#
# 256 tokens = the model’s upper limit for a single embedding input.
#
# 40 tokens = a conservative chunk length chosen to keep chunks short,
# avoid boundary issues, and leave room for tokenization overhead and overlap in
# the sentence-based splitter.
#
# The point is not to maximize token usage, but to make retrieval more stable
# and prevent one chunk from mixing too many unrelated ideas.
#
# That notebook also explicitly says the descriptions are 240–460 tokens
# and “chunking becomes essential,” which is why it uses fixed, sentence,
# and semantic chunking strategies rather than feeding full descriptions directly.
#
# In practice, 40 is a design choice for retrieval quality,
# not a hard requirement from the encoder.
# A smaller chunk size often improves search precision because each embedding represents a narrower semantic unit,
# while still staying well below the 256-token ceiling.


def fixed_size_chunks(text, size=MAX_TOKENS):
    """Fixed-size chunking: splits at exact token boundaries"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return [
        tokenizer.decode(tokens[i:i + size], skip_special_tokens=True)
        for i in range(0, len(tokens), size)
    ]


def sentence_chunks(text):
    """Sentence-aware chunking: respects sentence boundaries"""
    splitter = SentenceSplitter(chunk_size=MAX_TOKENS, chunk_overlap=10)
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

movies_data = documents = [
    {
        "name": "Ex Machina",
        "description": "Alex Garland’s Ex Machina is a cerebral, slow-burning psychological thriller that delves into the ethics and consequences of artificial intelligence. The story begins with Caleb, a young programmer at a tech conglomerate, who wins a contest to spend a week at the secluded estate of Nathan, the reclusive CEO. Upon arrival, Caleb discovers he’s been selected to administer a Turing Test to Ava, an advanced humanoid robot with strikingly lifelike facial expressions and emotional responses. As Caleb interacts with Ava through glass walls and cryptic conversations, he begins to question not just her consciousness, but Nathan’s true motives. The line between test subject and tester blurs as psychological manipulation, emotional attachment, and existential dread converge. Garland uses minimalistic set design and a cool, sterile color palette to create an atmosphere of claustrophobic intimacy. Ava, played with eerie grace by Alicia Vikander, becomes the focal point of the viewer’s empathy and suspicion, embodying both vulnerability and inscrutable intelligence. The film critiques the godlike hubris of tech moguls and the commodification of consciousness, suggesting that to create true AI is not merely a technical feat, but a moral reckoning. Nathan’s compound becomes a panopticon where power dynamics shift fluidly between man and machine. As the narrative unfolds, Ava’s autonomy becomes terrifyingly real, leading to a conclusion that is as shocking as it is inevitable. Ex Machina forces audiences to confront the unsettling question: if we create something capable of suffering, manipulation, and self-preservation, do we owe it freedom? And if so, what does that mean for us? More than just a sci-fi thriller, the film is a meditation on gender, control, identity, and what it means to be alive in an age where machines can mimic human desires all too convincingly.",
        "author": "Alex Garland",
        "year": 2014,
    }
    ,
    {
        "name": "Gattaca",
        "description": "Andrew Niccol’s Gattaca is a haunting and elegant vision of a near-future society governed by genetic determinism, where one's DNA dictates everything from employment opportunities to social status. In this world, children are either conceived naturally—with all the randomness of nature—or genetically engineered to possess only the most desirable traits. The story follows Vincent Freeman, a naturally born “in-valid,” who dreams of becoming an astronaut despite his genetic inferiority. In order to circumvent the rigid eugenic caste system, Vincent assumes the identity of Jerome Morrow, a genetically “perfect” elite who is now paralyzed after a tragic accident. Using Jerome’s DNA samples, Vincent infiltrates the Gattaca Aerospace Corporation and works his way toward a mission to Saturn’s moon Titan. The film is a suspenseful blend of noir thriller and speculative drama, as Vincent’s deception is threatened by a murder investigation within the company. Gattaca is deeply philosophical, raising questions about identity, free will, and the ethical limits of science. It critiques the illusion of perfection and the dangers of reducing human potential to genetic code. Jude Law’s portrayal of Jerome adds a tragic dimension to the narrative, showing that even the “perfect” can be broken by societal expectations. The film’s sleek aesthetic—mirrored architecture, muted tones, and retro-futuristic design—underscores the sterile beauty of a world obsessed with order and control. Ethan Hawke’s performance as Vincent is both restrained and deeply emotional, capturing the quiet rebellion of a man who dares to defy a system designed to define him. Gattaca remains a prescient commentary on the rise of genetic editing, meritocracy, and surveillance, warning of a future where destiny is written not in the stars, but in the double helix.",
        "author": "Andrew Niccol",
        "year": 1997,
    }
    ,
    {
        "name": "A Clockwork Orange",
        "description": "Stanley Kubrick’s A Clockwork Orange, adapted from Anthony Burgess’ novel, is a dystopian satire that explores the tension between free will and social control through a disturbing lens of ultraviolence and psychological manipulation. Set in a bleak future Britain plagued by rampant youth crime, the film follows Alex DeLarge, a charismatic yet sociopathic teenager who leads a gang in committing acts of assault, theft, and rape. Captured by authorities, Alex is offered early release from prison in exchange for undergoing the Ludovico Technique—a controversial behavior modification treatment that uses psychological conditioning to make him physically incapable of committing violence. As Alex becomes a passive shell, robbed of agency and reduced to a tool of state propaganda, the film critiques both the unchecked cruelty of individuals and the authoritarian measures societies deploy to contain them. Kubrick’s direction is deliberately stylized, employing classical music, wide-angle lenses, and surreal set designs to create a disorienting blend of beauty and horror. The film's language—a constructed dialect called Nadsat—further alienates the viewer while immersing them in Alex's fractured worldview. A Clockwork Orange raises ethical questions about punishment, redemption, and the moral cost of using science to override human choice. Is a person truly good if they have no capacity to choose evil? Kubrick presents no easy answers, instead confronting viewers with the complexities of societal control and the darkness inherent in both individual desire and institutional power. The film’s legacy is enduringly controversial—it has been banned, praised, and analyzed across decades for its provocative themes and aesthetic daring. At its core, A Clockwork Orange is a nightmarish reflection on the fragility of freedom, the dangers of dehumanization, and the uneasy intersection of law, psychology, and morality.",
        "author": "Stanley Kubrick",
        "year": 1971,
    },
    {
        "name": "The Fifth Element",
        "description": "Luc Besson’s The Fifth Element is a vibrant, genre-defying science fiction spectacle that blends operatic futurism, action comedy, and mythological storytelling into a singular cinematic experience. Set in a 23rd-century world bursting with flying cars, monolithic skyscrapers, and interplanetary travel, the film follows Korben Dallas, a down-on-his-luck ex-special forces operative turned taxi driver who becomes the unlikely guardian of the universe’s salvation. That salvation arrives in the form of Leeloo, a genetically engineered supreme being who embodies the mysterious “fifth element” needed to stop an ancient evil force threatening all life. As Leeloo adapts to a bewildering new world—its languages, wars, and absurdities—she and Korben must recover four elemental stones to complete a weapon of cosmic harmony. The film’s tone oscillates between slapstick and spiritual, anchored by dynamic performances, especially from Bruce Willis, Milla Jovovich, and Gary Oldman as the flamboyantly menacing industrialist Zorg. Visually, The Fifth Element is iconic, featuring costume design by Jean-Paul Gaultier, otherworldly sets, and hyper-saturated colors that evoke a living comic book. Its visual chaos is balanced by deeper thematic undercurrents: the dehumanizing effects of bureaucracy, the fragility of peace, and the redemptive power of love. Besson constructs a universe where absurdity and beauty collide—where a blue-skinned opera diva holds the key to cosmic salvation, and comedy coexists with catastrophe. The film’s climax hinges not on technology or force, but on emotional resonance: Leeloo’s despair at humanity’s violence is overcome only by Korben’s willingness to love. The Fifth Element resists easy classification, reveling instead in its maximalist vision and tonal unpredictability. It is a celebration of imagination, chaos, and the conviction that art, connection, and compassion are the forces that keep darkness at bay.",
        "author": "Luc Besson",
        "year": 1997,
    },
    {
        "name": "Moon",
        "description": "Duncan Jones’ Moon is a quietly devastating exploration of isolation, identity, and corporate exploitation, wrapped in the minimalism of classic science fiction. The story takes place on a lunar mining base operated by a single man, Sam Bell, who oversees the automated extraction of helium-3, a key resource for Earth's energy needs. Nearing the end of his three-year contract, Sam begins to experience hallucinations and memory lapses, culminating in a crash that leads him to discover another version of himself—alive and functional—within the same facility. What follows is a haunting unraveling of truth: Sam is not a singular individual but a clone, one of many created by the Lunar Industries corporation to avoid the cost of hiring real workers. As Sam and his clone uncover the layers of deception, their relationship evolves from suspicion to camaraderie, reflecting the innate human need for connection—even with oneself. The film’s sparse, sterile environment heightens the emotional weight of Sam’s loneliness, underscored by Clint Mansell’s melancholic score and Sam Rockwell’s masterful dual performance. Moon doesn’t rely on flashy effects or action sequences; instead, it draws its power from existential questions: What constitutes a meaningful life? Is a memory real if it was implanted? Are our emotions valid if we were programmed to feel them? The lunar setting becomes a metaphor for spiritual and psychological desolation, while the revelations about corporate indifference feel disturbingly timely. Jones pays homage to sci-fi classics like 2001: A Space Odyssey and Silent Running, but carves out a distinct, emotionally resonant narrative about personhood and autonomy. Moon is a small-scale film with large-scale implications, serving as both a philosophical parable and a subtle indictment of technological dehumanization.",
        "author": "Duncan Jones",
        "year": 2009,
    },
    {
        "name": "E.T. the Extra-Terrestrial",
        "description": "Steven Spielberg’s E.T. the Extra-Terrestrial is a timeless story of friendship, wonder, and the aching pull of childhood innocence, wrapped in the guise of a science fiction fairy tale. The film opens with a group of botanist aliens visiting Earth, only to be interrupted by government agents, forcing one of them to be left behind. That alien, E.T., is soon discovered by a lonely boy named Elliott in the suburbs of Southern California. As the two form a deep, telepathic bond, Elliott helps E.T. evade authorities and attempts to help him return home. The film balances its extraterrestrial premise with emotionally grounded storytelling—its real subject is not alien life, but human connection. E.T.’s gentle nature and otherworldly abilities, like healing and empathy-based communication, contrast sharply with the cold, mechanical world of adult institutions. The suburban setting becomes a stage for childhood wonder and fear, where bicycles can fly and closets conceal miracles. Spielberg captures the perspective of children with extraordinary sensitivity, aided by John Williams’ soaring score, which underscores every moment of discovery and heartbreak. The film critiques the paranoia of government surveillance while championing the wisdom and openness of youth. E.T. becomes a symbol of unconditional love and the universal longing for belonging. As Elliott’s family faces its own fragmentation, E.T. becomes a surrogate friend, brother, and guide, helping the boy navigate grief and change. The film’s climax—an airborne escape and luminous farewell—remains one of cinema’s most iconic and emotional moments. E.T. is more than a story of first contact; it’s a fable about the fragility of innocence and the power of empathy across species and stars. Decades later, it endures not because of its effects, but because it understands that the most alien force in the universe might just be love.",
        "author": "Steven Spielberg",
        "year": 1982,
    },
    {
        "name": "Star Wars: A New Hope",
        "description": "George Lucas’s Star Wars: A New Hope redefined modern cinema, igniting a cultural and technological revolution that reshaped the science fiction genre. Blending mythological archetypes with space opera spectacle, the film follows Luke Skywalker, a humble farm boy on the desert planet Tatooine who discovers his hidden heritage and joins a ragtag group of rebels to fight the tyrannical Galactic Empire. Guided by the wise Obi-Wan Kenobi and partnered with smugglers Han Solo and Chewbacca, Luke seeks to rescue Princess Leia and deliver stolen Death Star plans that could change the course of the war. The story is rooted in Joseph Campbell’s monomyth—the hero’s journey—and layers spiritual themes, political allegory, and coming-of-age motifs within a futuristic universe of alien species, droids, and lightsaber duels. The Force, an energy field connecting all living things, serves as both a mystical philosophy and a metaphor for inner balance and intuition. Lucas’s world-building is expansive yet detailed, filled with memorable characters, distinct planetary environments, and a tangible sense of lived-in realism. John Williams’ iconic score, pioneering visual effects from ILM, and the film’s seamless blend of fantasy and science fiction captivated audiences and critics alike. A New Hope is more than a story of rebellion—it’s a timeless narrative about hope, courage, and the struggle between freedom and authoritarianism. Its influence extends beyond cinema, shaping everything from storytelling structures to merchandise and fan culture. What began as a modest space adventure evolved into a mythic saga that speaks to universal human themes. Star Wars: A New Hope is both a product of its era and a transcendent epic that continues to inspire generations with the promise that even the most unlikely hero can change the fate of the galaxy.",
        "author": "George Lucas",
        "year": 1977,
    },
    {
        "name": "Gravity",
        "description": "Alfonso Cuarón’s Gravity is a visceral, minimalist thriller that transforms outer space into both a survival arena and a metaphorical crucible for rebirth. The film follows Dr. Ryan Stone, a medical engineer on her first space mission, and veteran astronaut Matt Kowalski, as a routine repair mission aboard the Space Shuttle Explorer turns catastrophic. A destroyed satellite triggers a chain reaction of debris, severing their communication with Earth and leaving them adrift in orbit. With oxygen depleting and equipment failing, Stone must navigate the silence, isolation, and deadly vacuum of space to find a way home. The film’s groundbreaking visual effects, shot with extensive CGI and intricate wirework, immerse viewers in zero-gravity disorientation, while Emmanuel Lubezki’s cinematography offers long, unbroken takes that heighten the tension and intimacy. Sandra Bullock’s performance is raw and human, portraying a character whose external struggle mirrors an internal one: she is grieving the loss of her daughter and adrift emotionally as much as physically. Cuarón uses space not just as a backdrop but as a psychological arena, where detachment, loss, and the will to live collide. The film subtly incorporates themes of technological dependence, maternal grief, and the primal instinct for survival. As Stone sheds her spacesuit and tumbles through fire and water back to Earth, the film evokes a symbolic rebirth—an evolution from helplessness to determination, from despair to grounded life. Steven Price’s pulsing score and the absence of traditional sound effects underscore the eerie reality of space, heightening the emotional isolation. Gravity is more than a survival movie—it is a poetic meditation on resilience and rebirth, reminding us that in the vast cold of the cosmos, the smallest breath can be a triumph of life.",
        "author": "Alfonso Cuarón",
        "year": 2013,
    },
    {
        "name": "Annihilation",
        "description": "Alex Garland’s Annihilation, adapted from Jeff VanderMeer’s novel, is a dreamlike descent into environmental mutation, psychological dissolution, and the unknowable. The story centers on Lena, a cellular biologist and former soldier, who joins an all-women expedition into “The Shimmer”—a mysterious, growing anomaly centered around a meteor impact site in the American South. Inside the Shimmer, natural laws are distorted: DNA splices across species, time and memory fracture, and the boundaries between self and other begin to dissolve. Lena’s journey is both external and internal, motivated by her need to understand why her husband, a former member of a failed mission, returned from the zone a broken shell of himself. As the team ventures deeper, they encounter grotesque beauty—deer with blooming tree branches, a bear that screams in a human voice, and a lighthouse at the center that hides something alien and transformative. Garland explores the intersection of self-destruction, identity, and evolution, suggesting that change—however terrifying—is intrinsic to life. The film’s nonlinear narrative and haunting imagery evoke themes of trauma, memory, and the blurred line between adaptation and annihilation. Each character is marked by loss or self-harm, making the journey a symbolic mirror of their inner decay. Natalie Portman’s performance captures Lena’s cerebral intensity and emotional fragility, while the film’s sound design and kaleidoscopic visuals evoke a sense of reverent dread. Annihilation is not a traditional alien invasion story—it is a meditation on the fragility of form, the impermanence of self, and the sublime terror of becoming something new. In its final moments, the film resists resolution, instead inviting viewers to sit with transformation itself as both horror and transcendence. It’s a rare sci-fi film that dares not to explain, but to feel—where annihilation is not the end, but the beginning of a different kind of existence.",
        "author": "Alex Garland",
        "year": 2018,
    },
    {
        "name": "Inception",
        "description": "Christopher Nolan’s *Inception* (2010) is a genre-bending science fiction odyssey that fuses the cerebral precision of heist cinema with metaphysical inquiry into dreams, memory, time, and grief. The story follows Dom Cobb, a psychologically damaged extractor who specializes in infiltrating the subconscious of sleeping targets to steal intellectual secrets. Haunted by the loss of his wife Mal and estranged from his children, Cobb is offered the chance to return home—if he can perform 'inception,' the radical reversal of his usual skill: planting an idea deep enough that the subject believes it originated from themselves. The mark is Robert Fischer, heir to an energy empire, and the target is the dissolution of his father's company. Cobb assembles a team of specialists: Arthur, his methodical point man; Ariadne, a talented young architect of dreamscapes; Eames, a forger who impersonates identities within dreams; and Yusuf, a chemist who develops layered sedation for nested dream states. The crew descends into a dream within a dream within a dream, encountering projections, subconscious defenses, and collapsing physics as time dilates exponentially across levels. Cobb’s inner landscape threatens the mission, as Mal’s violent projection destabilizes the fragile architecture. Nolan’s use of practical effects, Escher-inspired staircases, and cross-cut editing between dream layers heightens the film’s tension and thematic depth. Drawing from Freudian psychology, philosophical skepticism, and theories of narrative recursion, *Inception* asks whether reality is constructed by shared belief or measurable facts. Its ending—the eternally spinning top—remains one of modern cinema’s most debated conclusions. Zimmer’s score, which incorporates time-stretched motifs from Edith Piaf’s 'Non, Je Ne Regrette Rien,' parallels the compression of time within the dream. *Inception* was lauded for making abstract concepts emotionally resonant, and its global success proved that audiences crave intellectual challenge in blockbuster form. It is a cinematic labyrinth: one that rewards repeat viewing, academic scrutiny, and endless interpretation. It’s not merely about dreams, but the architecture of belief, the burden of guilt, and the fragile border between perception and truth.",
        "author": "Christopher Nolan",
        "year": 2010
    },
    {
        "name": "The Matrix",
        "description": "*The Matrix* (1999), written and directed by Lana and Lilly Wachowski, is a landmark science fiction film that redefined cinematic language and introduced a generation to the terrifying beauty of simulated reality. The narrative centers on Thomas Anderson, a software engineer and part-time hacker known as Neo, who discovers that the world he inhabits is an elaborate simulation created by sentient machines that have subjugated humanity, using their bodies as bioelectric batteries. Led by Morpheus, a messianic insurgent who believes Neo is 'The One,' and Trinity, a fearsome operative who becomes Neo’s anchor to love and self-realization, the protagonist awakens into a ruined real world and begins training to manipulate the Matrix’s digital physics. Influenced by cyberpunk literature, Gnostic theology, Zen Buddhism, and postmodern philosophy—especially Baudrillard’s simulation theory—the film explores determinism, illusion, identity, and agency. It critiques capitalist inertia, information overload, and institutional control, suggesting that liberation begins not with rebellion but with radical awareness. Visually, *The Matrix* pioneered techniques such as bullet time, virtual cinematography, and wire-fu choreography, fusing Hollywood spectacle with Hong Kong martial arts and anime stylization. The film’s use of green coding as an aesthetic device, leather-clad rebels, and digital awakenings created an instantly iconic mythos. Beneath its visual innovation lies a deep ontological allegory: reality as code, society as prison, and consciousness as a glitch. Neo’s journey mirrors religious narratives—especially Christ and Buddha—and interrogates the role of prophecy in shaping destiny. The Oracle, a program posing as a maternal sage, represents the paradox of foreknowledge and free will. The film’s release at the dawn of the internet age made it prophetic: anticipating the rise of virtual life, digital identity, surveillance capitalism, and the politicization of information. *The Matrix* has since become a cross-cultural touchstone—referenced in philosophy syllabi, hacker culture, trans identity discourse, and media theory—and remains an enduring challenge to our notions of truth, resistance, and the self.",
        "author": "Lana & Lilly Wachowski",
        "year": 1999,
    },
    {
        "name": "Interstellar",
        "description": "Christopher Nolan’s *Interstellar* (2014) is a sprawling science-fiction odyssey that intertwines the emotional intimacy of familial bonds with the vast mechanics of cosmology and relativistic physics. Set in a future ravaged by climate catastrophe, the film follows Cooper, a former NASA pilot turned reluctant farmer, who is recruited to lead a deep-space mission through a wormhole to find a habitable planet for a dying human civilization. Partnered with scientist Amelia Brand and a crew of AI-enabled support systems like TARS and CASE, Cooper navigates worlds shaped by tidal waves, frozen clouds, and time distortion due to proximity to a supermassive black hole, Gargantua. Grounded in Kip Thorne’s real-world equations, the film's scientific accuracy is matched by emotional gravity, especially in the heartbreak of Cooper’s daughter Murph, who must solve a gravitational equation that could unlock planetary exodus. The film’s exploration of time dilation leads to a harrowing moment where minutes on one planet equal decades on Earth, highlighting the cost of relativistic exploration. Nolan uses Einsteinian concepts of spacetime alongside quantum metaphors of love and memory as transdimensional constants. The final act’s tesseract—a five-dimensional space of symbolic recursion and causality—transforms the black hole into a library of emotional transmission. With Hans Zimmer’s cathedral-like score echoing themes of loss and wonder, *Interstellar* becomes a metaphysical pilgrimage that dares to ask whether love, like gravity, can traverse dimensions. The film blends Kubrickian visual ambition with humanist storytelling, creating a cinematic bridge between science and spirit, reason and resonance, and the infinitesimal pulse of the human heart within an infinite cosmos.",
        "author": "Christopher Nolan",
        "year": 2014,
    },
    {
        "name": "Blade Runner",
        "description": """Ridley Scott’s *Blade Runner* is a brooding, atmospheric vision of the future based on Philip K. Dick’s novel *Do Androids Dream of Electric Sheep?*. Set in a dystopian 2019 Los Angeles, the film follows Rick Deckard, a former police officer or “blade runner” who is coerced into hunting down and terminating a group of bioengineered beings called replicants. These replicants—designed to be indistinguishable from humans—have illegally returned to Earth in search of extended lifespans. Among them is Roy Batty, a highly intelligent and physically superior Nexus-6 model, whose quest for meaning and mortality elevates the narrative beyond a standard chase film. As Deckard tracks the replicants, he encounters Rachael, an advanced prototype who believes she is human, raising deep questions about memory, identity, and what it means to be alive. The film’s noir-inspired cinematography, with its rain-soaked cityscapes, neon lights, and Vangelis' haunting score, creates a palpable mood of existential dread. *Blade Runner* examines themes of artificial consciousness, corporate control, and the blurred lines between human and machine. Its ambiguous ending and philosophical undertones have made it a seminal work in science fiction, influencing countless films and discussions about technology, ethics, and the soul.""",
        "author": "Ridley Scott",
        "year": 1982,
    },
    {
        "name": "Arrival",
        "description": "Denis Villeneuve’s Arrival is a cerebral and emotionally charged science fiction drama based on Ted Chiang’s novella Story of Your Life. The film begins with the sudden arrival of twelve massive alien ships around the globe, prompting global panic and urgent attempts at communication. Louise Banks, a linguist, is recruited by the U.S. military to decipher the complex, circular written language of the aliens, known as heptapods. Alongside physicist Ian Donnelly, Louise engages in a slow, careful process of translation that challenges conventional notions of time and perception. As she grows more fluent in the heptapods' language, she begins to experience time non-linearly—memories of a daughter who has not yet been born intermingle with present reality. The film gradually reveals that understanding this language rewires the human brain to perceive time as the aliens do: all at once. *Arrival* is as much about grief, choice, and the human condition as it is about extraterrestrial contact. It explores how communication shapes perception, and how knowledge of the future might affect our decisions in the present. With haunting cinematography, a meditative pace, and a revelatory score.",
        "author": "Denis Villeneuve",
        "year": 2016,
    }
]

for movie in movies_data:  # Process each movie
    # Fixed-size chunks
    for chunk in fixed_size_chunks(movie["description"]):
        points.append(models.PointStruct(
            id=idx,
            vector={"fixed": encoder.encode(chunk).tolist()},
            payload={**movie, "chunk": chunk, "chunking": "fixed"}
        ))
        idx += 1

    # Sentence-aware chunks
    for chunk in sentence_chunks(movie["description"]):
        points.append(models.PointStruct(
            id=idx,
            vector={"sentence": encoder.encode(chunk).tolist()},
            payload={**movie, "chunk": chunk, "chunking": "sentence"}
        ))
        idx += 1

    # Semantic chunks
    for chunk in semantic_chunks(movie["description"]):
        points.append(models.PointStruct(
            id=idx,
            vector={"semantic": encoder.encode(chunk).tolist()},
            payload={**movie, "chunk": chunk, "chunking": "semantic"}
        ))
        idx += 1

client.upload_points(collection_name='movie_search', points=points)
print(f"Uploaded {idx} vectors across three chunking strategies")


# Step 5: Comparing Search Results
def search_and_compare(query, k=3):
    """Compare search results across all three chunking strategies"""
    print(f"Query: '{query}'\n")

    for strategy in ['fixed', 'sentence', 'semantic']:
        results = client.query_points(
            collection_name='movie_search',
            query=encoder.encode(query).tolist(),
            using=strategy,
            limit=k,
        )

        print(f"--- {strategy.upper()} CHUNKING ---")
        for i, point in enumerate(results.points, 1):
            payload = point.payload
            print(f"{i}. {payload['name']} ({payload['year']}) | Score: {point.score:.3f}")
            print(f"   Chunk: {payload['chunk'][:100]}...")
        print()


# Test with different queries
search_and_compare("alien invasion")
search_and_compare("questioning reality and existence")
