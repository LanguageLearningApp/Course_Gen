# components.py
components = [
    {"name": "Flashcards", "skill": "Memorization", "difficulty": 1, "word_count": "Flexible"},
    {"name": "Matching", "skill": "Memorization", "difficulty": 2, "word_count": 4},
    {"name": "Memory", "skill": "Memorization", "difficulty": 3, "word_count": "3-5"},
    {"name": "Conversation", "skill": "Speaking", "difficulty": 9, "word_count": "N/A"},
    {"name": "Item Describe", "skill": "Reasoning", "difficulty": 10, "word_count": "N/A"},
    {"name": "Fill in the blank", "skill": "Familiarity", "difficulty": 3, "word_count": 1},
    {"name": "Passage", "skill": "Reading", "difficulty": 5, "word_count": "N/A"},
    {"name": "Multiple Choice", "skill": "Memorization", "difficulty": 3, "word_count": 4},
    {"name": "Pronunciation", "skill": "Memorization", "difficulty": 3, "word_count": 1},
    {"name": "Photolist", "skill": "Memorization", "difficulty": 5, "word_count": "N/A"},
    {"name": "Trivia", "skill": "Game", "difficulty": 4, "word_count": "N/A"},
    {"name": "Word Fall", "skill": "Memorization", "difficulty": 4, "word_count": "3-7"},
    {"name": "Item Description", "skill": "Reasoning", "difficulty": 8, "word_count": 1},
    {"name": "Sentence Translation (Easy)", "skill": "Familiarity", "difficulty": 3, "word_count": "N/A"},
    {"name": "Sentence Translation (Medium)", "skill": "Familiarity", "difficulty": 4, "word_count": "N/A"},
    {"name": "Sentence Translation (Hard)", "skill": "Familiarity", "difficulty": 6, "word_count": "N/A"},
    {"name": "Listening Recap (Coming Soon)", "skill": "Listening", "difficulty": 8, "word_count": "N/A"},
    {"name": "Sound Translation - Word (Coming Soon)", "skill": "Listening", "difficulty": 3, "word_count": 1},
    {"name": "Sound Translation - Sentence (Coming Soon)", "skill": "Listening", "difficulty": 5, "word_count": 1},
]

sections = [
    {
        "name": "Food",
        "words": [
            "manzana", "pan", "leche", "queso", "huevo",  # apple, bread, milk, cheese, egg
            "carne", "pollo", "pescado", "arroz", "frijoles",  # meat, chicken, fish, rice, beans
            "sopa", "ensalada", "fruta", "verdura", "pastel"  # soup, salad, fruit, vegetable, cake
        ],
        "performance": 0.85
    },
    {
        "name": "Places",
        "words": [
            "casa", "escuela", "parque", "tienda", "hospital",  # house, school, park, shop, hospital
            "biblioteca", "cine", "restaurante", "museo", "oficina",  # library, cinema, restaurant, museum, office
            "iglesia", "estación", "aeropuerto", "plaza", "mercado"  # church, station, airport, square, market
        ],
        "performance": 0.95
    },
    {
        "name": "Animals",
        "words": [
            "gato", "perro", "pájaro", "pez", "caballo",  # cat, dog, bird, fish, horse
            "vaca", "oveja", "cerdo", "gallina", "conejo",  # cow, sheep, pig, hen, rabbit
            "tigre", "elefante", "serpiente", "mono", "oso"  # tiger, elephant, snake, monkey, bear
        ],
        "performance": 1.0
    },
    {
        "name": "Nature",
        "words": [
            "árbol", "flor", "río", "montaña", "sol",  # tree, flower, river, mountain, sun
            "luna", "estrella", "nube", "lluvia", "viento",  # moon, star, cloud, rain, wind
            "lago", "bosque", "playa", "desierto", "cielo"  # lake, forest, beach, desert, sky
        ],
        "performance": 1.0
    },
    {
        "name": "Travel",
        "words": [
            "coche", "avión", "tren", "barco", "bicicleta",  # car, airplane, train, boat, bicycle
            "autobús", "taxi", "camión", "moto", "maleta",  # bus, taxi, truck, motorcycle, suitcase
            "boleto", "mapa", "hotel", "pasaporte", "viaje"  # ticket, map, hotel, passport, trip
        ],
        "performance": 1.0
    },
    {
        "name": "Clothing",
        "words": [
            "camisa", "pantalones", "zapatos", "sombrero", "chaqueta",  # shirt, pants, shoes, hat, jacket
            "bufanda", "guantes", "falda", "vestido", "calcetines",  # scarf, gloves, skirt, dress, socks
            "cinturón", "abrigo", "bolsa", "gafas", "sudadera"  # belt, coat, bag, glasses, sweatshirt
        ],
        "performance": 0.90
    },
    {
        "name": "Family",
        "words": [
            "madre", "padre", "hermano", "hermana", "abuelo",  # mother, father, brother, sister, grandfather
            "abuela", "tío", "tía", "primo", "prima",  # grandmother, uncle, aunt, male cousin, female cousin
            "hijo", "hija", "esposo", "esposa", "sobrino"  # son, daughter, husband, wife, nephew
        ],
        "performance": 0.88
    },
    {
        "name": "Colors",
        "words": [
            "rojo", "azul", "verde", "amarillo", "negro",  # red, blue, green, yellow, black
            "blanco", "naranja", "morado", "rosa", "gris",  # white, orange, purple, pink, gray
            "marrón", "dorado", "plateado", "turquesa", "violeta"  # brown, gold, silver, turquoise, violet
        ],
        "performance": 1.0
    },
    {
        "name": "Body",
        "words": [
            "cabeza", "brazo", "pierna", "mano", "pie",  # head, arm, leg, hand, foot
            "ojo", "nariz", "boca", "oreja", "dedo",  # eye, nose, mouth, ear, finger
            "cuello", "espalda", "hombro", "rodilla", "corazón"  # neck, back, shoulder, knee, heart
        ],
        "performance": 0.92
    },
    {
        "name": "Time",
        "words": [
            "día", "noche", "mañana", "tarde", "hora",  # day, night, morning, afternoon, hour
            "minuto", "segundo", "semana", "mes", "año",  # minute, second, week, month, year
            "lunes", "viernes", "ayer", "hoy", "mañana"  # Monday, Friday, yesterday, today, tomorrow
        ],
        "performance": 1.0
    },
    {
        "name": "Weather",
        "words": [
            "sol", "lluvia", "nieve", "viento", "nube",  # sun, rain, snow, wind, cloud
            "tormenta", "trueno", "relámpago", "niebla", "hielo",  # storm, thunder, lightning, fog, ice
            "calor", "frío", "húmedo", "seco", "clima"  # heat, cold, humid, dry, weather
        ],
        "performance": 0.87
    },
    {
        "name": "Furniture",
        "words": [
            "mesa", "silla", "sofá", "cama", "armario",  # table, chair, sofa, bed, wardrobe
            "escritorio", "estante", "lámpara", "alfombra", "espejo",  # desk, shelf, lamp, carpet, mirror
            "cajón", "puerta", "ventana", "televisor", "sillón"  # drawer, door, window, television, armchair
        ],
        "performance": 1.0
    },
    {
        "name": "School",
        "words": [
            "libro", "lápiz", "cuaderno", "bolígrafo", "mochila",  # book, pencil, notebook, pen, backpack
            "pizarra", "tiza", "borrador", "regla", "tijeras",  # blackboard, chalk, eraser, ruler, scissors
            "maestro", "alumno", "clase", "examen", "escuela"  # teacher, student, class, exam, school
        ],
        "performance": 0.93
    },
    {
        "name": "Sports",
        "words": [
            "fútbol", "baloncesto", "tenis", "natación", "ciclismo",  # soccer, basketball, tennis, swimming, cycling
            "pelota", "raqueta", "gol", "cancha", "jugador",  # ball, racket, goal, court, player
            "corredor", "gimnasio", "trofeo", "equipo", "deporte"  # runner, gym, trophy, team, sport
        ],
        "performance": 1.0
    },
    {
        "name": "Jobs",
        "words": [
            "médico", "profesor", "ingeniero", "abogado", "cocinero",  # doctor, teacher, engineer, lawyer, cook
            "policía", "bombero", "carpintero", "vendedor", "escritor",  # police, firefighter, carpenter, salesman, writer
            "artista", "músico", "actor", "jardinero", "piloto"  # artist, musician, actor, gardener, pilot
        ],
        "performance": 0.89
    },
    {
        "name": "Emotions",
        "words": [
            "feliz", "triste", "enojado", "cansado", "sorprendido",  # happy, sad, angry, tired, surprised
            "asustado", "contento", "nervioso", "calmado", "aburrido",  # scared, pleased, nervous, calm, bored
            "orgulloso", "avergonzado", "alegre", "confundido", "emocionado"  # proud, embarrassed, cheerful, confused, excited
        ],
        "performance": 1.0
    },
    {
        "name": "Technology",
        "words": [
            "computadora", "teléfono", "televisión", "radio", "cámara",  # computer, phone, television, radio, camera
            "internet", "correo", "pantalla", "teclado", "ratón",  # internet, email, screen, keyboard, mouse
            "cable", "batería", "auriculares", "impresora", "robot"  # cable, battery, headphones, printer, robot
        ],
        "performance": 0.91
    },
    {
        "name": "Music",
        "words": [
            "guitarra", "piano", "violín", "tambor", "flauta",  # guitar, piano, violin, drum, flute
            "cantante", "canción", "nota", "ritmo", "melodía",  # singer, song, note, rhythm, melody
            "concierto", "orquesta", "disco", "volumen", "baila"  # concert, orchestra, record, volume, dance
        ],
        "performance": 1.0
    },
    {
        "name": "Health",
        "words": [
            "doctor", "enfermera", "medicina", "pastilla", "hospital",  # doctor, nurse, medicine, pill, hospital
            "dolor", "fiebre", "tos", "gripe", "sangre",  # pain, fever, cough, flu, blood
            "salud", "ejercicio", "dieta", "vitaminas", "cura"  # health, exercise, diet, vitamins, cure
        ],
        "performance": 0.86
    },
    {
        "name": "Hobbies",
        "words": [
            "leer", "escribir", "pintar", "cocinar", "bailar",  # read, write, paint, cook, dance
            "cantar", "jugar", "caminar", "pescar", "fotografía",  # sing, play, walk, fish, photography
            "colección", "jardín", "viajar", "cine", "deportes"  # collection, garden, travel, movies, sports
        ],
        "performance": 1.0
    },
    # -------------------- Additional 20 Sections --------------------
    {
        "name": "Kitchen",
        "words": [
            "cuchillo", "tenedor", "cuchara", "sartén", "olla",  # knife, fork, spoon, pan, pot
            "plato", "vaso", "taza", "microondas", "horno",  # plate, glass, cup, microwave, oven
            "estufa", "batidora", "colador", "tabla", "exprimidor"  # stove, blender, colander, cutting board, juicer
        ],
        "performance": 1.0
    },
    {
        "name": "Office",
        "words": [
            "escritorio", "silla", "computadora", "impresora", "papel",  # desk, chair, computer, printer, paper
            "bolígrafo", "portapapeles", "archivo", "calculadora", "teléfono",  # pen, clipboard, file, calculator, phone
            "agenda", "sujetapapeles", "lámpara", "estantería", "reunión"  # planner, paperclip, lamp, shelf, meeting
        ],
        "performance": 1.0
    },
    {
        "name": "Wild Animals",
        "words": [
            "león", "cebra", "jirafa", "elefante", "rinoceronte",  # lion, zebra, giraffe, elephant, rhinoceros
            "hipopótamo", "cocodrilo", "águila", "búho", "lobo",  # hippopotamus, crocodile, eagle, owl, wolf
            "oso polar", "puma", "gacela", "guepardo", "búfalo"  # polar bear, puma, gazelle, cheetah, buffalo
        ],
        "performance": 1.0
    },
    {
        "name": "Flowers",
        "words": [
            "rosa", "margarita", "tulipán", "lirio", "orquídea",  # rose, daisy, tulip, lily, orchid
            "girasol", "dalia", "violeta", "jazmín", "petunia",  # sunflower, dahlia, violet, jasmine, petunia
            "clavel", "loto", "magnolia", "hortensia", "azucena"  # carnation, lotus, magnolia, hydrangea, Madonna lily
        ],
        "performance": 1.0
    },
    {
        "name": "Insects",
        "words": [
            "mariposa", "abeja", "hormiga", "mosca", "saltamontes",  # butterfly, bee, ant, fly, grasshopper
            "grillo", "avispa", "chinche", "cucaracha", "escarabajo",  # cricket, wasp, bug, cockroach, beetle
            "libélula", "cochinilla", "mosquito", "termita", "pulga"  # dragonfly, mealybug, mosquito, termite, flea
        ],
        "performance": 1.0
    },
    {
        "name": "Shapes",
        "words": [
            "círculo", "cuadrado", "triángulo", "rectángulo", "ovalo",  # circle, square, triangle, rectangle, oval
            "rombo", "estrella", "corazón", "espiral", "zigzag",  # diamond, star, heart, spiral, zigzag
            "luna creciente", "hexágono", "octágono", "pentágono", "paralelogramo"  # crescent, hexagon, octagon, pentagon, parallelogram
        ],
        "performance": 1.0
    },
    {
        "name": "Verbs",
        "words": [
            "ser", "estar", "tener", "hacer", "poder",  # to be, to be, to have, to do, can
            "decir", "ir", "ver", "dar", "saber",  # say, go, see, give, know
            "querer", "llegar", "pasar", "deber", "poner"  # want, arrive, pass, owe, put
        ],
        "performance": 1.0
    },
    {
        "name": "Adjectives",
        "words": [
            "bueno", "malo", "grande", "pequeño", "feliz",  # good, bad, big, small, happy
            "triste", "rápido", "lento", "fácil", "difícil",  # sad, fast, slow, easy, difficult
            "caliente", "frío", "nuevo", "viejo", "interesante"  # hot, cold, new, old, interesting
        ],
        "performance": 1.0
    },
    {
        "name": "Adverbs",
        "words": [
            "rápidamente", "lentamente", "bien", "mal", "aquí",  # quickly, slowly, well, badly, here
            "allí", "siempre", "nunca", "hoy", "ayer",  # there, always, never, today, yesterday
            "mañana", "ahora", "después", "antes", "fuera"  # tomorrow, now, later, before, outside
        ],
        "performance": 1.0
    },
    {
        "name": "Prepositions",
        "words": [
            "a", "ante", "bajo", "cabe", "con",  # to, before, under, next to, with
            "contra", "de", "desde", "en", "entre",  # against, of, from, in, among
            "hacia", "hasta", "para", "por", "según"  # toward, until, for, by, according to
        ],
        "performance": 1.0
    },
    {
        "name": "Conjunctions",
        "words": [
            "y", "o", "pero", "aunque", "sino",  # and, or, but, although, rather
            "pues", "porque", "mientras", "además", "sin embargo",  # since, because, while, furthermore, however
            "no obstante", "así que", "ya que", "cuando", "donde"  # nevertheless, so, since, when, where
        ],
        "performance": 1.0
    },
    {
        "name": "Family Relations",
        "words": [
            "abuelo", "abuela", "tío", "tía", "primo",  # grandfather, grandmother, uncle, aunt, cousin
            "prima", "sobrino", "sobrina", "suegro", "suegra",  # cousin, nephew, niece, father-in-law, mother-in-law
            "cuñado", "cuñada", "nieto", "nieta", "padrastro"  # brother-in-law, sister-in-law, grandson, granddaughter, stepfather
        ],
        "performance": 1.0
    },
    {
        "name": "Occupations",
        "words": [
            "doctor", "enfermero", "maestro", "ingeniero", "arquitecto",  # doctor, nurse, teacher, engineer, architect
            "abogado", "contable", "periodista", "electricista", "plomero",  # lawyer, accountant, journalist, electrician, plumber
            "cocinero", "piloto", "soldado", "policía", "agricultor"  # cook, pilot, soldier, police, farmer
        ],
        "performance": 1.0
    },
    {
        "name": "Technology & Gadgets",
        "words": [
            "smartphone", "tablet", "laptop", "router", "disco duro",  # smartphone, tablet, laptop, router, hard drive
            "memoria", "software", "hardware", "cámara digital", "auriculares",  # memory, software, hardware, digital camera, headphones
            "monitor", "teclado", "mouse", "impresora", "drone"  # monitor, keyboard, mouse, printer, drone
        ],
        "performance": 1.0
    },
    {
        "name": "Transportation",
        "words": [
            "carretera", "autopista", "señal", "semáforo", "pasajero",  # road, highway, sign, traffic light, passenger
            "conductor", "estación", "terminal", "billete", "vía",  # driver, station, terminal, ticket, track
            "tren", "metro", "carril", "peatón", "bache"  # train, subway, lane, pedestrian, pothole
        ],
        "performance": 1.0
    },
    {
        "name": "Geography",
        "words": [
            "montaña", "río", "valle", "desierto", "isla",  # mountain, river, valley, desert, island
            "continente", "océano", "península", "costa", "llanura",  # continent, ocean, peninsula, coast, plain
            "vulcán", "glaciar", "bosque", "selva", "laguna"  # volcano, glacier, forest, jungle, lagoon
        ],
        "performance": 1.0
    },
    {
        "name": "History",
        "words": [
            "historia", "revolución", "guerra", "imperio", "civilización",  # history, revolution, war, empire, civilization
            "reinado", "batalla", "descubrimiento", "colonización", "independencia",  # reign, battle, discovery, colonization, independence
            "reforma", "dictadura", "monarquía", "república", "acontecimiento"  # reform, dictatorship, monarchy, republic, event
        ],
        "performance": 1.0
    },
    {
        "name": "Science",
        "words": [
            "ciencia", "experimento", "teoría", "hipótesis", "física",  # science, experiment, theory, hypothesis, physics
            "química", "biología", "astronomía", "matemáticas", "laboratorio",  # chemistry, biology, astronomy, mathematics, laboratory
            "descubrimiento", "energía", "átomo", "molécula", "genética"  # discovery, energy, atom, molecule, genetics
        ],
        "performance": 1.0
    },
    {
        "name": "Art",
        "words": [
            "arte", "pintura", "escultura", "dibujo", "collage",  # art, painting, sculpture, drawing, collage
            "mural", "performance", "instalación", "grafito", "óleo",  # mural, performance, installation, graphite, oil
            "acuarela", "boceto", "exposición", "estilo", "abstracto"  # watercolor, sketch, exhibition, style, abstract
        ],
        "performance": 1.0
    },
    {
        "name": "Literature",
        "words": [
            "novela", "cuento", "poesía", "drama", "ensayo",  # novel, short story, poetry, drama, essay
            "prosa", "ficción", "sátira", "tragedia", "comedia",  # prose, fiction, satire, tragedy, comedy
            "epopeya", "microrrelato", "romance", "autobiografía", "diario"  # epic, flash fiction, romance, autobiography, diary
        ],
        "performance": 1.0
    }
]
