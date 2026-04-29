import numpy as np
from mda.core.bind import DIM, normalize

CONCEPT_SEEDS = {
    "COMPILE":       100, "RUN":          101, "DEVELOP":      102,
    "LEARN":         103, "USE":           104, "BUILD":        105,
    "ANALYZE":       106, "PROCESS":       107, "CALCULATE":    108,
    "EXECUTE":       109, "DEPLOY":        110, "TEST":         111,
    "DEBUG":         112, "OPTIMIZE":      113, "REFACTOR":     114,
    "INSTALL":       115, "CONFIGURE":     116, "MONITOR":      117,
    "STORE":         118, "RETRIEVE":      119, "TRANSFORM":    120,
    "GENERATE":      121, "PREDICT":       122, "CLASSIFY":     123,
    "DETECT":        124, "SEARCH":        125, "FILTER":       126,
    "SORT":          127, "MERGE":         128, "SPLIT":        129,
    "ENCODE":        130, "DECODE":        131, "PARSE":        132,
    "RENDER":        133, "STREAM":        134, "CACHE":        135,
    "MIGRATE":       136, "SCALE":         137, "SECURE":       138,
    "AUTHENTICATE":  139, "AUTHORIZE":     140, "ENCRYPT":      141,
    "COMPRESS":      142, "SERIALIZE":     143, "VALIDATE":     144,
    "SIMULATE":      145, "VISUALIZE":     146, "TRAIN":        147,
    "INFER":         148, "EMBED":         149, "TOKENIZE":     150,

    "FAST":          200, "SLOW":          201, "STRONG":       202,
    "SAFE":          203, "FLEXIBLE":      204, "POPULAR":      205,
    "MODULAR":       206, "EFFICIENT":     207, "READABLE":     208,
    "SCALABLE":      209, "RELIABLE":      210, "PORTABLE":     211,
    "LIGHTWEIGHT":   212, "POWERFUL":      213, "SIMPLE":       214,
    "COMPLEX":       215, "DYNAMIC":       216, "STATIC":       217,
    "ASYNC":         218, "PARALLEL":      219, "DISTRIBUTED":  220,
    "REAL_TIME":     221, "OFFLINE":       222, "OPEN_SOURCE":  223,
    "ENTERPRISE":    224, "EXPERIMENTAL":  225, "STABLE":       226,
    "LEGACY":        227, "MODERN":        228, "IMMUTABLE":    229,
    "MUTABLE":       230, "TYPED":         231, "UNTYPED":      232,
    "COMPILED":      233, "INTERPRETED":   234, "FUNCTIONAL":   235,
    "IMPERATIVE":    236, "DECLARATIVE":   237, "REACTIVE":     238,
    "CONCURRENT":    239, "DETERMINISTIC": 240, "PROBABILISTIC":241,
    "LOSSLESS":      242, "LOSSY":         243, "SYMMETRIC":    244,
    "ASYMMETRIC":    245, "LINEAR":        246, "NONLINEAR":    247,

    "SOFTWARE":      300, "LANGUAGE":      301, "LIBRARY":      302,
    "FRAMEWORK":     303, "DATABASE":      304, "SYSTEM":       305,
    "ALGORITHM":     306, "PROTOCOL":      307, "API":          308,
    "INTERFACE":     309, "ARCHITECTURE":  310, "PATTERN":      311,
    "NETWORK":       312, "SERVER":        313, "CLIENT":       314,
    "CONTAINER":     315, "CLUSTER":       316, "PIPELINE":     317,
    "REPOSITORY":    318, "PACKAGE":       319, "MODULE":       320,
    "CLASS":         321, "FUNCTION":      322, "VARIABLE":     323,
    "LOOP":          324, "CONDITION":     325, "EXCEPTION":    326,
    "THREAD":        327, "PROCESS_OBJ":   328, "MEMORY":       329,
    "STACK":         330, "QUEUE":         331, "TREE":         332,
    "GRAPH":         333, "HASH":          334, "ARRAY":        335,
    "MATRIX":        336, "TENSOR":        337, "VECTOR_OBJ":   338,
    "STREAM_OBJ":    339, "BUFFER":        340, "SOCKET":       341,
    "TOKEN":         342, "PARSER":        343, "COMPILER":     344,
    "RUNTIME":       345, "KERNEL":        346, "SHELL":        347,
    "TERMINAL":      348, "BROWSER":       349, "EDITOR":       350,

    "AI":            400, "MODEL":         401, "VECTOR":       402,
    "TRAINING":      403, "NEURAL":        404, "GRADIENT":     405,
    "LOSS":          406, "WEIGHT":        407, "LAYER":        408,
    "ATTENTION":     409, "EMBEDDING":     410, "FINE_TUNE":    411,
    "INFERENCE":     412, "BACKPROP":      413, "OVERFITTING":  414,
    "REGULARIZE":    415, "DROPOUT":       416, "BATCH":        417,
    "EPOCH":         418, "OPTIMIZER":     419, "ACTIVATION":   420,
    "CONVOLUTION":   421, "POOLING":       422, "NORMALIZATION":423,
    "TRANSFORMER":   424, "ENCODER_OBJ":   425, "DECODER_OBJ":  426,
    "DIFFUSION":     427, "GENERATIVE":    428, "DISCRIMINATIVE":429,
    "SUPERVISED":    430, "UNSUPERVISED":  431, "REINFORCEMENT":432,
    "REWARD":        433, "POLICY":        434, "AGENT":        435,
    "ENVIRONMENT":   436, "CLUSTERING":    437, "REGRESSION":   438,
    "CLASSIFICATION":439, "SEGMENTATION":  440, "DETECTION":    441,
    "GENERATION":    442, "TRANSLATION":   443, "SUMMARIZATION":444,
    "TOKENIZATION":  445, "VOCABULARY":    446, "CONTEXT":      447,
    "PROMPT":        448, "CHAIN":         449,

    "DATA":          500, "SCIENCE":       501, "STATISTICS":   502,
    "PROBABILITY":   503, "CALCULUS":      504, "ALGEBRA":      505,
    "GEOMETRY":      506, "TOPOLOGY":      507, "LOGIC":        508,
    "SET":           509, "FUNCTION_MATH": 510, "INTEGRAL":     511,
    "DERIVATIVE":    512, "TRANSFORM":     513, "FOURIER":      514,
    "EIGENVALUE":    515, "DETERMINANT":   516, "DISTRIBUTION": 517,
    "HYPOTHESIS":    518, "CORRELATION":   519, "VARIANCE":     520,

    "PHYSICS":       600, "CHEMISTRY":     601, "BIOLOGY":      602,
    "QUANTUM":       603, "RELATIVITY":    604, "THERMODYNAMICS":605,
    "ENERGY":        606, "FORCE":         607, "MASS":         608,
    "WAVE":          609, "PARTICLE":      610, "ATOM":         611,
    "MOLECULE":      612, "CELL":          613, "DNA":          614,
    "PROTEIN":       615, "EVOLUTION":     616, "NEURON":       617,
    "SYNAPSE":       618, "GENOME":        619,

    "HUMAN":         700, "BRAIN":         701, "HEART":        702,
    "LANGUAGE_NAT":  703, "MEMORY_COG":    704, "EMOTION":      705,
    "CONSCIOUSNESS": 706, "LEARNING_COG":  707, "SOCIAL":       708,
    "ECONOMIC":      709, "LEGAL":         710, "POLITICAL":    711,
    "HISTORICAL":    712, "CULTURAL":      713, "ARTISTIC":     714,
    "CITY":          715, "COUNTRY":       716, "GEOGRAPHY":    717,
    "CLIMATE":       718, "ANIMAL":        719, "PLANT":        720,

    "RUN_SPORT":     800, "TIRED":         801, "REST":         802,
    "SPORT":         803, "COMPETITION":   804, "TEAM":         805,
    "VICTORY":       806, "DEFEAT":        807,

    "ERROR":         900, "WARNING":       901, "SUCCESS":      902,
    "FAILURE":       903, "TIMEOUT":       904, "RETRY":        905,
    "ROLLBACK":      906, "BACKUP":        907, "RESTORE":      908,
    "LOG":           909, "METRIC":        910, "ALERT":        911,
}

TR_CONCEPT_MAP = {
    "derlenir": "COMPILE",      "derler": "COMPILE",        "derlendi": "COMPILE",      "derlemek": "COMPILE",
    "çalışır":  "RUN",          "çalıştı": "RUN",           "çalışıyor": "RUN",         "çalıştırır": "EXECUTE",
    "geliştirir": "DEVELOP",    "geliştirdi": "DEVELOP",    "geliştirme": "DEVELOP",    "geliştirildi": "DEVELOP",
    "öğrenir":  "LEARN",        "öğrendi": "LEARN",         "öğrenme": "LEARN",         "öğretir": "LEARN",
    "kullanır":  "USE",         "kullandı": "USE",          "kullanım": "USE",          "kullanılır": "USE",
    "kurar":    "BUILD",        "kurdu": "BUILD",           "kurulum": "BUILD",         "inşa": "BUILD",
    "analiz":   "ANALYZE",      "analiz eder": "ANALYZE",   "çözümler": "ANALYZE",
    "işler":    "PROCESS",      "işledi": "PROCESS",        "işlem": "PROCESS",
    "hesaplar": "CALCULATE",    "hesaplama": "CALCULATE",   "hesapladı": "CALCULATE",
    "çalıştırır": "EXECUTE",    "yürütür": "EXECUTE",       "yürütme": "EXECUTE",
    "dağıtır":  "DEPLOY",       "yayınlar": "DEPLOY",       "deploy": "DEPLOY",
    "test eder": "TEST",        "test": "TEST",             "sınar": "TEST",
    "hata ayıklar": "DEBUG",    "debug": "DEBUG",           "hata ayıklama": "DEBUG",
    "optimize eder": "OPTIMIZE","optimize": "OPTIMIZE",     "iyileştirir": "OPTIMIZE",
    "yeniden düzenler": "REFACTOR", "refactor": "REFACTOR",
    "yükler":   "INSTALL",      "kurulur": "INSTALL",       "yükleme": "INSTALL",
    "yapılandırır": "CONFIGURE","ayarlar": "CONFIGURE",     "konfigüre": "CONFIGURE",
    "izler":    "MONITOR",      "takip eder": "MONITOR",    "monitoring": "MONITOR",
    "depolar":  "STORE",        "saklar": "STORE",          "kaydeder": "STORE",
    "alır":     "RETRIEVE",     "getirir": "RETRIEVE",      "çeker": "RETRIEVE",
    "dönüştürür": "TRANSFORM",  "çevirir": "TRANSFORM",     "transform": "TRANSFORM",
    "üretir":   "GENERATE",     "oluşturur": "GENERATE",    "generate": "GENERATE",
    "tahmin eder": "PREDICT",   "öngörür": "PREDICT",       "tahmin": "PREDICT",
    "sınıflandırır": "CLASSIFY","kategorize": "CLASSIFY",   "sınıflar": "CLASSIFY",
    "algılar":  "DETECT",       "tespit eder": "DETECT",    "tespit": "DETECT",
    "arar":     "SEARCH",       "sorgular": "SEARCH",       "arama": "SEARCH",
    "filtreler": "FILTER",      "filtre": "FILTER",
    "sıralar":  "SORT",         "sıralama": "SORT",
    "birleştirir": "MERGE",     "birleşme": "MERGE",
    "böler":    "SPLIT",        "bölme": "SPLIT",
    "kodlar":   "ENCODE",       "şifreler": "ENCODE",
    "çözer":    "DECODE",       "şifre çözer": "DECODE",
    "ayrıştırır": "PARSE",      "parse": "PARSE",
    "görüntüler": "RENDER",     "render": "RENDER",
    "akış":     "STREAM",       "stream": "STREAM",
    "önbellek": "CACHE",        "cache": "CACHE",
    "taşır":    "MIGRATE",      "migration": "MIGRATE",
    "ölçekler": "SCALE",        "ölçeklenir": "SCALE",      "scale": "SCALE",
    "güvence altına alır": "SECURE", "korur": "SECURE",
    "kimlik doğrular": "AUTHENTICATE", "doğrular": "AUTHENTICATE",
    "yetkilendirir": "AUTHORIZE","yetki": "AUTHORIZE",
    "şifreler": "ENCRYPT",      "şifreleme": "ENCRYPT",
    "sıkıştırır": "COMPRESS",   "sıkıştırma": "COMPRESS",
    "serileştirir": "SERIALIZE","serialize": "SERIALIZE",
    "doğrular": "VALIDATE",     "validation": "VALIDATE",
    "simüle eder": "SIMULATE",  "simülasyon": "SIMULATE",
    "görselleştirir": "VISUALIZE","vizualize": "VISUALIZE",
    "eğitir":   "TRAIN",        "eğitim": "TRAINING",       "eğitilir": "TRAIN",
    "çıkarım":  "INFER",        "inference": "INFER",       "çıkarım yapar": "INFER",
    "gömer":    "EMBED",        "gömme": "EMBED",           "embedding": "EMBED",
    "tokenize": "TOKENIZE",     "token": "TOKEN",           "tokenlar": "TOKEN",

    "hızlı":    "FAST",         "hızlıdır": "FAST",         "hız": "FAST",              "süratli": "FAST",
    "yavaş":    "SLOW",         "yavaştır": "SLOW",
    "güçlü":    "STRONG",       "güçlüdür": "STRONG",       "kuvvetli": "STRONG",
    "güvenli":  "SAFE",         "güvenlidir": "SAFE",       "güvenlik": "SAFE",         "emniyetli": "SAFE",
    "esnek":    "FLEXIBLE",     "esnektir": "FLEXIBLE",     "uyarlanabilir": "FLEXIBLE",
    "popüler":  "POPULAR",      "yaygın": "POPULAR",        "tercih edilen": "POPULAR",
    "modüler":  "MODULAR",      "modülerdir": "MODULAR",
    "verimli":  "EFFICIENT",    "verimliliği": "EFFICIENT", "etkin": "EFFICIENT",
    "okunabilir": "READABLE",   "okunaklı": "READABLE",
    "ölçeklenebilir": "SCALABLE","ölçeklenir": "SCALABLE",
    "güvenilir": "RELIABLE",    "güvenilirdir": "RELIABLE",
    "taşınabilir": "PORTABLE",  "platformdan bağımsız": "PORTABLE",
    "hafif":    "LIGHTWEIGHT",  "hafiftir": "LIGHTWEIGHT",
    "güçlü":    "POWERFUL",     "kuvvetli": "POWERFUL",
    "basit":    "SIMPLE",       "sade": "SIMPLE",
    "karmaşık": "COMPLEX",      "karmaşıktır": "COMPLEX",
    "dinamik":  "DYNAMIC",      "dinamiktir": "DYNAMIC",
    "statik":   "STATIC",       "statiktir": "STATIC",
    "asenkron": "ASYNC",        "async": "ASYNC",           "eşzamansız": "ASYNC",
    "paralel":  "PARALLEL",     "eşzamanlı": "PARALLEL",
    "dağıtık":  "DISTRIBUTED",  "dağıtılmış": "DISTRIBUTED",
    "gerçek zamanlı": "REAL_TIME","anlık": "REAL_TIME",
    "açık kaynak": "OPEN_SOURCE","özgür yazılım": "OPEN_SOURCE",
    "kurumsal": "ENTERPRISE",   "enterprise": "ENTERPRISE",
    "kararlı":  "STABLE",       "stabil": "STABLE",
    "modern":   "MODERN",       "güncel": "MODERN",
    "derlenen": "COMPILED",     "derlenir": "COMPILED",
    "yorumlanan": "INTERPRETED","yorumlanır": "INTERPRETED",
    "fonksiyonel": "FUNCTIONAL","işlevsel": "FUNCTIONAL",
    "nesne yönelimli": "IMPERATIVE","oop": "IMPERATIVE",
    "bildirimsel": "DECLARATIVE","declarative": "DECLARATIVE",
    "reaktif":  "REACTIVE",     "tepkisel": "REACTIVE",
    "eşzamanlı": "CONCURRENT",  "concurrent": "CONCURRENT",
    "olasılıksal": "PROBABILISTIC","stokastik": "PROBABILISTIC",

    "yazılım":  "SOFTWARE",     "yazılımdır": "SOFTWARE",   "software": "SOFTWARE",
    "dil":      "LANGUAGE",     "dilidir": "LANGUAGE",      "programlama": "LANGUAGE",  "programlama dili": "LANGUAGE",
    "kütüphane": "LIBRARY",     "kütüphanesidir": "LIBRARY","lib": "LIBRARY",
    "çerçeve":  "FRAMEWORK",    "framework": "FRAMEWORK",
    "veritabanı": "DATABASE",   "database": "DATABASE",     "veri tabanı": "DATABASE",
    "sistem":   "SYSTEM",       "sistemdir": "SYSTEM",
    "algoritma": "ALGORITHM",   "algorithm": "ALGORITHM",
    "protokol": "PROTOCOL",     "protocol": "PROTOCOL",
    "arayüz":   "INTERFACE",    "interface": "INTERFACE",   "api": "API",
    "mimari":   "ARCHITECTURE", "architecture": "ARCHITECTURE",
    "desen":    "PATTERN",      "pattern": "PATTERN",
    "ağ":       "NETWORK",      "network": "NETWORK",
    "sunucu":   "SERVER",       "server": "SERVER",
    "istemci":  "CLIENT",       "client": "CLIENT",
    "konteyner": "CONTAINER",   "container": "CONTAINER",   "docker": "CONTAINER",
    "küme":     "CLUSTER",      "cluster": "CLUSTER",
    "boru hattı": "PIPELINE",   "pipeline": "PIPELINE",
    "depo":     "REPOSITORY",   "repo": "REPOSITORY",
    "paket":    "PACKAGE",      "package": "PACKAGE",
    "modül":    "MODULE",       "module": "MODULE",
    "sınıf":    "CLASS",        "class": "CLASS",
    "fonksiyon": "FUNCTION",    "function": "FUNCTION",     "metod": "FUNCTION",
    "değişken": "VARIABLE",     "variable": "VARIABLE",
    "döngü":    "LOOP",         "loop": "LOOP",
    "koşul":    "CONDITION",    "if": "CONDITION",
    "istisna":  "EXCEPTION",    "exception": "EXCEPTION",   "hata": "ERROR",
    "iş parçacığı": "THREAD",   "thread": "THREAD",
    "bellek":   "MEMORY",       "memory": "MEMORY",         "ram": "MEMORY",
    "yığın":    "STACK",        "stack": "STACK",
    "kuyruk":   "QUEUE",        "queue": "QUEUE",
    "ağaç":     "TREE",         "tree": "TREE",
    "grafik":   "GRAPH",        "graph": "GRAPH",
    "hash":     "HASH",         "karma": "HASH",
    "dizi":     "ARRAY",        "liste": "ARRAY",
    "matris":   "MATRIX",       "matrix": "MATRIX",
    "tensör":   "TENSOR",       "tensor": "TENSOR",
    "vektör":   "VECTOR",       "vector": "VECTOR",         "vektörlerle": "VECTOR",
    "akış":     "STREAM_OBJ",   "stream": "STREAM_OBJ",
    "tampon":   "BUFFER",       "buffer": "BUFFER",
    "soket":    "SOCKET",       "socket": "SOCKET",
    "derleyici": "COMPILER",    "compiler": "COMPILER",
    "çalışma zamanı": "RUNTIME","runtime": "RUNTIME",
    "çekirdek": "KERNEL",       "kernel": "KERNEL",
    "kabuk":    "SHELL",        "shell": "SHELL",
    "tarayıcı": "BROWSER",      "browser": "BROWSER",

    "yapay":    "AI",           "zeka": "AI",               "yapay zeka": "AI",         "ai": "AI",
    "model":    "MODEL",        "models": "MODEL",
    "gradient": "GRADIENT",     "gradyan": "GRADIENT",
    "kayıp":    "LOSS",         "loss": "LOSS",             "kayıp fonksiyonu": "LOSS",
    "ağırlık":  "WEIGHT",       "weight": "WEIGHT",         "parametre": "WEIGHT",
    "katman":   "LAYER",        "layer": "LAYER",
    "dikkat":   "ATTENTION",    "attention": "ATTENTION",   "self-attention": "ATTENTION",
    "gömme":    "EMBEDDING",    "embedding": "EMBEDDING",
    "ince ayar": "FINE_TUNE",   "fine-tune": "FINE_TUNE",   "finetune": "FINE_TUNE",
    "çıkarım":  "INFERENCE",    "inference": "INFERENCE",
    "geri yayılım": "BACKPROP", "backprop": "BACKPROP",
    "aşırı öğrenme": "OVERFITTING","overfitting": "OVERFITTING",
    "düzenlileştirme": "REGULARIZE","regularization": "REGULARIZE",
    "bırakma":  "DROPOUT",      "dropout": "DROPOUT",
    "toplu":    "BATCH",        "batch": "BATCH",
    "dönem":    "EPOCH",        "epoch": "EPOCH",
    "optimize edici": "OPTIMIZER","optimizer": "OPTIMIZER",
    "aktivasyon": "ACTIVATION", "activation": "ACTIVATION",
    "evrişim":  "CONVOLUTION",  "convolution": "CONVOLUTION","cnn": "CONVOLUTION",
    "normalizasyon": "NORMALIZATION","normalization": "NORMALIZATION",
    "dönüştürücü": "TRANSFORMER","transformer": "TRANSFORMER",
    "difüzyon":  "DIFFUSION",   "diffusion": "DIFFUSION",
    "üretken":  "GENERATIVE",   "generative": "GENERATIVE", "gan": "GENERATIVE",
    "denetimli": "SUPERVISED",  "supervised": "SUPERVISED",
    "denetimsiz": "UNSUPERVISED","unsupervised": "UNSUPERVISED",
    "pekiştirmeli": "REINFORCEMENT","reinforcement": "REINFORCEMENT","rl": "REINFORCEMENT",
    "ödül":     "REWARD",       "reward": "REWARD",
    "politika": "POLICY",       "policy": "POLICY",
    "kümeleme":  "CLUSTERING",  "clustering": "CLUSTERING",
    "regresyon": "REGRESSION",  "regression": "REGRESSION",
    "segmentasyon": "SEGMENTATION","segmentation": "SEGMENTATION",
    "bağlam":   "CONTEXT",      "context": "CONTEXT",
    "istem":    "PROMPT",       "prompt": "PROMPT",

    "veri":     "DATA",         "verisi": "DATA",           "data": "DATA",
    "bilim":    "SCIENCE",      "bilimsel": "SCIENCE",
    "istatistik": "STATISTICS", "istatistiksel": "STATISTICS",
    "olasılık": "PROBABILITY",  "probability": "PROBABILITY",
    "kalkülüs": "CALCULUS",     "türev": "DERIVATIVE",      "integral": "INTEGRAL",
    "lineer cebir": "ALGEBRA",  "matris": "MATRIX",
    "dağılım":  "DISTRIBUTION", "distribution": "DISTRIBUTION",
    "korelasyon": "CORRELATION","correlation": "CORRELATION",
    "varyans":  "VARIANCE",     "variance": "VARIANCE",
    "hipotez":  "HYPOTHESIS",   "hypothesis": "HYPOTHESIS",

    "fizik":    "PHYSICS",      "kuantum": "QUANTUM",       "görelilik": "RELATIVITY",
    "termodinamik": "THERMODYNAMICS","enerji": "ENERGY",    "kuvvet": "FORCE",
    "dalga":    "WAVE",         "parçacık": "PARTICLE",     "atom": "ATOM",
    "molekül":  "MOLECULE",     "kimya": "CHEMISTRY",
    "biyoloji": "BIOLOGY",      "hücre": "CELL",            "dna": "DNA",
    "protein":  "PROTEIN",      "evrim": "EVOLUTION",       "nöron": "NEURON",
    "sinaps":   "SYNAPSE",      "genom": "GENOME",

    "insan":    "HUMAN",        "insandır": "HUMAN",        "beyin": "BRAIN",
    "kalp":     "HEART",        "dil":      "LANGUAGE_NAT", "hafıza": "MEMORY_COG",
    "bellek":   "MEMORY_COG",   "duygu": "EMOTION",         "bilinç": "CONSCIOUSNESS",
    "öğrenme":  "LEARNING_COG", "sosyal": "SOCIAL",         "ekonomi": "ECONOMIC",
    "hukuk":    "LEGAL",        "siyaset": "POLITICAL",     "tarih": "HISTORICAL",
    "kültür":   "CULTURAL",     "sanat": "ARTISTIC",        "şehir": "CITY",
    "ülke":     "COUNTRY",      "coğrafya": "GEOGRAPHY",    "iklim": "CLIMATE",
    "hayvan":   "ANIMAL",       "bitki": "PLANT",

    "koştu":    "RUN_SPORT",    "koşar": "RUN_SPORT",       "koşuyor": "RUN_SPORT",
    "maraton":  "RUN_SPORT",    "antrenman": "RUN_SPORT",   "yoruldu": "TIRED",
    "yorgun":   "TIRED",        "dinlendi": "REST",         "dinlenir": "REST",
    "spor":     "SPORT",        "sporcu": "SPORT",          "yarışma": "COMPETITION",
    "takım":    "TEAM",         "zafer": "VICTORY",         "yenilgi": "DEFEAT",

    "hata":     "ERROR",        "uyarı": "WARNING",         "başarı": "SUCCESS",
    "başarısız": "FAILURE",     "zaman aşımı": "TIMEOUT",   "yeniden dene": "RETRY",
    "geri al":  "ROLLBACK",     "yedek": "BACKUP",          "geri yükle": "RESTORE",
    "günlük":   "LOG",          "metrik": "METRIC",         "uyarı": "ALERT",
}

EN_CONCEPT_MAP = {
    "compiled":  "COMPILE",     "compiles": "COMPILE",      "compile": "COMPILE",
    "runs":      "RUN",         "run": "RUN",               "running": "RUN",
    "develops":  "DEVELOP",     "developed": "DEVELOP",     "development": "DEVELOP",
    "learns":    "LEARN",       "learned": "LEARN",         "learning": "LEARN",
    "uses":      "USE",         "used": "USE",              "usage": "USE",
    "builds":    "BUILD",       "built": "BUILD",           "building": "BUILD",
    "analyzes":  "ANALYZE",     "analysis": "ANALYZE",      "analyzing": "ANALYZE",
    "processes": "PROCESS",     "processed": "PROCESS",     "processing": "PROCESS",
    "calculates":"CALCULATE",   "calculation": "CALCULATE",
    "executes":  "EXECUTE",     "execution": "EXECUTE",     "execute": "EXECUTE",
    "deploys":   "DEPLOY",      "deployment": "DEPLOY",     "deploy": "DEPLOY",
    "tests":     "TEST",        "testing": "TEST",          "test": "TEST",
    "debugs":    "DEBUG",       "debugging": "DEBUG",       "debug": "DEBUG",
    "optimizes": "OPTIMIZE",    "optimization": "OPTIMIZE", "optimize": "OPTIMIZE",
    "refactors": "REFACTOR",    "refactoring": "REFACTOR",  "refactor": "REFACTOR",
    "installs":  "INSTALL",     "installation": "INSTALL",  "install": "INSTALL",
    "configures":"CONFIGURE",   "configuration": "CONFIGURE","config": "CONFIGURE",
    "monitors":  "MONITOR",     "monitoring": "MONITOR",    "monitor": "MONITOR",
    "stores":    "STORE",       "storage": "STORE",         "store": "STORE",
    "retrieves": "RETRIEVE",    "retrieval": "RETRIEVE",    "fetch": "RETRIEVE",
    "transforms":"TRANSFORM",   "transformation": "TRANSFORM","transform": "TRANSFORM",
    "generates": "GENERATE",    "generation": "GENERATE",   "generate": "GENERATE",
    "predicts":  "PREDICT",     "prediction": "PREDICT",    "predict": "PREDICT",
    "classifies":"CLASSIFY",    "classification": "CLASSIFY","classify": "CLASSIFY",
    "detects":   "DETECT",      "detection": "DETECT",      "detect": "DETECT",
    "searches":  "SEARCH",      "search": "SEARCH",         "query": "SEARCH",
    "filters":   "FILTER",      "filtering": "FILTER",      "filter": "FILTER",
    "sorts":     "SORT",        "sorting": "SORT",          "sort": "SORT",
    "merges":    "MERGE",       "merging": "MERGE",         "merge": "MERGE",
    "splits":    "SPLIT",       "splitting": "SPLIT",       "split": "SPLIT",
    "encodes":   "ENCODE",      "encoding": "ENCODE",       "encode": "ENCODE",
    "decodes":   "DECODE",      "decoding": "DECODE",       "decode": "DECODE",
    "parses":    "PARSE",       "parsing": "PARSE",         "parse": "PARSE",
    "renders":   "RENDER",      "rendering": "RENDER",      "render": "RENDER",
    "streams":   "STREAM",      "streaming": "STREAM",      "stream": "STREAM",
    "caches":    "CACHE",       "caching": "CACHE",         "cache": "CACHE",
    "migrates":  "MIGRATE",     "migration": "MIGRATE",     "migrate": "MIGRATE",
    "scales":    "SCALE",       "scaling": "SCALE",         "scale": "SCALE",
    "secures":   "SECURE",      "security": "SECURE",       "secure": "SECURE",
    "authenticates":"AUTHENTICATE","authentication": "AUTHENTICATE","auth": "AUTHENTICATE",
    "authorizes":"AUTHORIZE",   "authorization": "AUTHORIZE","authz": "AUTHORIZE",
    "encrypts":  "ENCRYPT",     "encryption": "ENCRYPT",    "encrypt": "ENCRYPT",
    "compresses":"COMPRESS",    "compression": "COMPRESS",  "compress": "COMPRESS",
    "serializes":"SERIALIZE",   "serialization": "SERIALIZE","serialize": "SERIALIZE",
    "validates": "VALIDATE",    "validation": "VALIDATE",   "validate": "VALIDATE",
    "simulates": "SIMULATE",    "simulation": "SIMULATE",   "simulate": "SIMULATE",
    "visualizes":"VISUALIZE",   "visualization": "VISUALIZE","visualize": "VISUALIZE",
    "trains":    "TRAIN",       "training": "TRAINING",     "train": "TRAIN",
    "infers":    "INFER",       "inference": "INFER",       "infer": "INFER",
    "embeds":    "EMBED",       "embedding": "EMBED",       "embed": "EMBED",
    "tokenizes": "TOKENIZE",    "tokenization": "TOKENIZE", "tokenize": "TOKENIZE",

    "fast":      "FAST",        "quick": "FAST",            "rapid": "FAST",            "speed": "FAST",
    "slow":      "SLOW",        "slowly": "SLOW",
    "strong":    "STRONG",      "powerful": "POWERFUL",     "robust": "STRONG",
    "safe":      "SAFE",        "secure": "SAFE",           "safety": "SAFE",
    "flexible":  "FLEXIBLE",    "adaptable": "FLEXIBLE",
    "popular":   "POPULAR",     "widely": "POPULAR",        "common": "POPULAR",
    "modular":   "MODULAR",     "modularized": "MODULAR",
    "efficient": "EFFICIENT",   "efficiency": "EFFICIENT",  "performant": "EFFICIENT",
    "readable":  "READABLE",    "clean": "READABLE",
    "scalable":  "SCALABLE",    "scalability": "SCALABLE",
    "reliable":  "RELIABLE",    "reliability": "RELIABLE",  "stable": "STABLE",
    "portable":  "PORTABLE",    "cross-platform": "PORTABLE",
    "lightweight":"LIGHTWEIGHT","minimal": "LIGHTWEIGHT",
    "simple":    "SIMPLE",      "easy": "SIMPLE",           "straightforward": "SIMPLE",
    "complex":   "COMPLEX",     "complicated": "COMPLEX",
    "dynamic":   "DYNAMIC",     "dynamically": "DYNAMIC",
    "static":    "STATIC",      "statically": "STATIC",
    "async":     "ASYNC",       "asynchronous": "ASYNC",    "non-blocking": "ASYNC",
    "parallel":  "PARALLEL",    "concurrent": "CONCURRENT", "multi-threaded": "PARALLEL",
    "distributed":"DISTRIBUTED","decentralized": "DISTRIBUTED",
    "real-time": "REAL_TIME",   "realtime": "REAL_TIME",    "live": "REAL_TIME",
    "open-source":"OPEN_SOURCE","opensource": "OPEN_SOURCE","free": "OPEN_SOURCE",
    "enterprise":"ENTERPRISE",  "production": "ENTERPRISE",
    "modern":    "MODERN",      "cutting-edge": "MODERN",   "latest": "MODERN",
    "compiled":  "COMPILED",    "interpreted": "INTERPRETED",
    "functional":"FUNCTIONAL",  "object-oriented": "IMPERATIVE","oop": "IMPERATIVE",
    "declarative":"DECLARATIVE","reactive": "REACTIVE",
    "probabilistic":"PROBABILISTIC","stochastic": "PROBABILISTIC",

    "software":  "SOFTWARE",    "program": "SOFTWARE",
    "language":  "LANGUAGE",    "programming": "LANGUAGE",  "lang": "LANGUAGE",
    "library":   "LIBRARY",     "libraries": "LIBRARY",     "lib": "LIBRARY",
    "framework": "FRAMEWORK",   "frameworks": "FRAMEWORK",
    "database":  "DATABASE",    "db": "DATABASE",           "datastore": "DATABASE",
    "system":    "SYSTEM",      "systems": "SYSTEM",
    "algorithm": "ALGORITHM",   "algo": "ALGORITHM",
    "protocol":  "PROTOCOL",    "spec": "PROTOCOL",
    "api":       "API",         "interface": "INTERFACE",   "sdk": "API",
    "architecture":"ARCHITECTURE","design": "ARCHITECTURE",
    "pattern":   "PATTERN",     "paradigm": "PATTERN",
    "network":   "NETWORK",     "net": "NETWORK",
    "server":    "SERVER",      "backend": "SERVER",        "host": "SERVER",
    "client":    "CLIENT",      "frontend": "CLIENT",
    "container": "CONTAINER",   "docker": "CONTAINER",      "pod": "CONTAINER",
    "cluster":   "CLUSTER",     "kubernetes": "CLUSTER",
    "pipeline":  "PIPELINE",    "workflow": "PIPELINE",
    "repository":"REPOSITORY",  "repo": "REPOSITORY",       "registry": "REPOSITORY",
    "package":   "PACKAGE",     "module": "MODULE",         "dependency": "PACKAGE",
    "class":     "CLASS",       "object": "CLASS",
    "function":  "FUNCTION",    "method": "FUNCTION",       "func": "FUNCTION",
    "variable":  "VARIABLE",    "var": "VARIABLE",
    "loop":      "LOOP",        "iteration": "LOOP",
    "condition": "CONDITION",   "branch": "CONDITION",
    "exception": "EXCEPTION",   "error": "ERROR",           "bug": "ERROR",
    "thread":    "THREAD",      "goroutine": "THREAD",
    "memory":    "MEMORY",      "ram": "MEMORY",            "heap": "MEMORY",
    "stack":     "STACK",       "call stack": "STACK",
    "queue":     "QUEUE",       "fifo": "QUEUE",
    "tree":      "TREE",        "binary tree": "TREE",
    "graph":     "GRAPH",       "dag": "GRAPH",
    "hash":      "HASH",        "hashmap": "HASH",          "dictionary": "HASH",
    "array":     "ARRAY",       "list": "ARRAY",            "slice": "ARRAY",
    "matrix":    "MATRIX",      "grid": "MATRIX",
    "tensor":    "TENSOR",      "nd-array": "TENSOR",
    "vector":    "VECTOR",      "vectors": "VECTOR",        "embedding": "EMBEDDING",
    "token":     "TOKEN",       "tokens": "TOKEN",
    "compiler":  "COMPILER",    "transpiler": "COMPILER",
    "runtime":   "RUNTIME",     "vm": "RUNTIME",            "jvm": "RUNTIME",
    "kernel":    "KERNEL",      "os": "KERNEL",
    "shell":     "SHELL",       "bash": "SHELL",            "cli": "SHELL",
    "browser":   "BROWSER",     "chrome": "BROWSER",

    "artificial":"AI",          "intelligence": "AI",       "ml": "AI",
    "model":     "MODEL",       "models": "MODEL",
    "gradient":  "GRADIENT",    "grad": "GRADIENT",
    "loss":      "LOSS",        "cost": "LOSS",
    "weight":    "WEIGHT",      "weights": "WEIGHT",        "parameter": "WEIGHT",
    "layer":     "LAYER",       "layers": "LAYER",
    "attention": "ATTENTION",   "self-attention": "ATTENTION","multi-head": "ATTENTION",
    "fine-tune": "FINE_TUNE",   "finetune": "FINE_TUNE",    "fine-tuning": "FINE_TUNE",
    "backprop":  "BACKPROP",    "backpropagation": "BACKPROP",
    "overfitting":"OVERFITTING","overfit": "OVERFITTING",
    "regularization":"REGULARIZE","regularize": "REGULARIZE","l2": "REGULARIZE",
    "dropout":   "DROPOUT",     "drop": "DROPOUT",
    "batch":     "BATCH",       "mini-batch": "BATCH",
    "epoch":     "EPOCH",       "epochs": "EPOCH",
    "optimizer": "OPTIMIZER",   "adam": "OPTIMIZER",        "sgd": "OPTIMIZER",
    "activation":"ACTIVATION",  "relu": "ACTIVATION",       "sigmoid": "ACTIVATION",
    "convolution":"CONVOLUTION","conv": "CONVOLUTION",       "cnn": "CONVOLUTION",
    "normalization":"NORMALIZATION","batchnorm": "NORMALIZATION","layernorm": "NORMALIZATION",
    "transformer":"TRANSFORMER","bert": "TRANSFORMER",       "gpt": "TRANSFORMER",
    "diffusion": "DIFFUSION",   "denoising": "DIFFUSION",
    "generative":"GENERATIVE",  "gan": "GENERATIVE",        "vae": "GENERATIVE",
    "supervised":"SUPERVISED",  "labeled": "SUPERVISED",
    "unsupervised":"UNSUPERVISED","unlabeled": "UNSUPERVISED","clustering": "CLUSTERING",
    "reinforcement":"REINFORCEMENT","rl": "REINFORCEMENT",   "policy": "POLICY",
    "reward":    "REWARD",      "agent": "AGENT",           "environment": "ENVIRONMENT",
    "regression":"REGRESSION",  "segmentation": "SEGMENTATION",
    "context":   "CONTEXT",     "window": "CONTEXT",
    "prompt":    "PROMPT",      "instruction": "PROMPT",

    "data":      "DATA",        "dataset": "DATA",          "corpus": "DATA",
    "science":   "SCIENCE",     "scientific": "SCIENCE",
    "statistics":"STATISTICS",  "statistical": "STATISTICS","stats": "STATISTICS",
    "probability":"PROBABILITY","probabilistic": "PROBABILITY",
    "calculus":  "CALCULUS",    "derivative": "DERIVATIVE", "integral": "INTEGRAL",
    "algebra":   "ALGEBRA",     "linear": "LINEAR",         "matrix": "MATRIX",
    "distribution":"DISTRIBUTION","gaussian": "DISTRIBUTION","normal": "DISTRIBUTION",
    "correlation":"CORRELATION","covariance": "CORRELATION",
    "variance":  "VARIANCE",    "std": "VARIANCE",
    "hypothesis":"HYPOTHESIS",  "test": "TEST",

    "physics":   "PHYSICS",     "quantum": "QUANTUM",       "relativity": "RELATIVITY",
    "thermodynamics":"THERMODYNAMICS","energy": "ENERGY",    "force": "FORCE",
    "wave":      "WAVE",        "particle": "PARTICLE",     "atom": "ATOM",
    "molecule":  "MOLECULE",    "chemistry": "CHEMISTRY",
    "biology":   "BIOLOGY",     "cell": "CELL",             "dna": "DNA",
    "protein":   "PROTEIN",     "evolution": "EVOLUTION",   "neuron": "NEURON",
    "synapse":   "SYNAPSE",     "genome": "GENOME",

    "human":     "HUMAN",       "person": "HUMAN",          "brain": "BRAIN",
    "heart":     "HEART",       "memory": "MEMORY_COG",     "emotion": "EMOTION",
    "consciousness":"CONSCIOUSNESS","social": "SOCIAL",      "economic": "ECONOMIC",
    "legal":     "LEGAL",       "political": "POLITICAL",   "historical": "HISTORICAL",
    "cultural":  "CULTURAL",    "artistic": "ARTISTIC",     "city": "CITY",
    "country":   "COUNTRY",     "geography": "GEOGRAPHY",   "climate": "CLIMATE",
    "animal":    "ANIMAL",      "plant": "PLANT",

    "ran":       "RUN_SPORT",   "marathon": "RUN_SPORT",    "tired": "TIRED",
    "fatigue":   "TIRED",       "rested": "REST",           "rest": "REST",
    "sport":     "SPORT",       "sports": "SPORT",          "competition": "COMPETITION",
    "team":      "TEAM",        "victory": "VICTORY",       "defeat": "DEFEAT",

    "error":     "ERROR",       "warning": "WARNING",       "success": "SUCCESS",
    "failure":   "FAILURE",     "timeout": "TIMEOUT",       "retry": "RETRY",
    "rollback":  "ROLLBACK",    "backup": "BACKUP",         "restore": "RESTORE",
    "log":       "LOG",         "metric": "METRIC",         "alert": "ALERT",
}

TR_SUFFIXES = [
    "dığımızdan", "tığımızdan", "ydıklarında",
    "mektedir", "maktadır", "lebilir", "labilir",
    "ıyor", "iyor", "uyor", "üyor",
    "acak", "ecek", "arak", "erek",
    "dır", "dir", "dur", "dür", "tır", "tir", "tur", "tür",
    "dan", "den", "tan", "ten",
    "nın", "nin", "nun", "nün",
    "lar", "ler", "lık", "lik",
    "dı", "di", "du", "dü", "tı", "ti", "tu", "tü",
    "ar", "er", "ir", "ır", "ur", "ür",
    "mı", "mi", "mu", "mü",
    "da", "de", "ta", "te",
    "ın", "in", "un", "ün",
    "ı", "i", "u", "ü", "a", "e",
]

STOPWORDS = {
    "bu", "bir", "ve", "ile", "de", "da", "mi", "mı",
    "için", "olan", "olarak", "gibi", "kadar", "daha",
    "a", "an", "the", "is", "are", "in", "of", "to",
    "for", "and", "or", "but", "with", "as", "at",
}


def _build_concept_vectors(dim: int) -> dict[str, np.ndarray]:
    vecs = {}
    for concept, seed in CONCEPT_SEEDS.items():
        vecs[concept] = normalize(
            np.random.default_rng(seed).normal(0, 1, dim)
        )
    return vecs


_STEM_CACHE: dict[str, str] = {}


def _stem_tr(word: str) -> str:
    try:
        return _STEM_CACHE[word]
    except KeyError:
        pass
    if len(word) <= 3:
        _STEM_CACHE[word] = word
        return word
    for suffix in TR_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 2:
            result = word[:-len(suffix)]
            _STEM_CACHE[word] = result
            return result
    _STEM_CACHE[word] = word
    return word


def _trigrams(word: str, n: int = 3) -> list[str]:
    if len(word) < n:
        return [word]
    return [word[i:i+n] for i in range(len(word) - n + 1)]


class HolisticEncoder:
    def __init__(self, dim: int = DIM):
        self.dim          = dim
        self._concepts    = _build_concept_vectors(dim)
        self._tri_vecs: dict[str, np.ndarray] = {}
        self._encode_cache: dict[str, np.ndarray] = {}
        self._build_trigram_index()

    def _build_trigram_index(self) -> None:
        all_words = list(TR_CONCEPT_MAP.keys()) + list(EN_CONCEPT_MAP.keys())
        for word in all_words:
            for tri in _trigrams(word):
                if tri not in self._tri_vecs:
                    seed = sum(ord(c) * (i+1) for i, c in enumerate(tri))
                    self._tri_vecs[tri] = normalize(
                        np.random.default_rng(seed % (2**31)).normal(0, 1, self.dim)
                    )

    def _normalize_text(self, text: str) -> str:
        text = text.strip().lower()
        for p in "?!.,;:\"'()[]{}":
            text = text.replace(p, " ")
        while "  " in text:
            text = text.replace("  ", " ")
        return text.strip()

    def _word_to_concept(self, word: str) -> np.ndarray | None:
        # Guard against stale keys in the global maps that point to concepts
        # registered on a different encoder instance
        for mapping in (TR_CONCEPT_MAP, EN_CONCEPT_MAP):
            if word in mapping:
                key = mapping[word]
                if key in self._concepts:
                    return self._concepts[key]
        stem = _stem_tr(word)
        for mapping in (TR_CONCEPT_MAP, EN_CONCEPT_MAP):
            if stem in mapping:
                key = mapping[stem]
                if key in self._concepts:
                    return self._concepts[key]
        return None

    def encode(self, text: str) -> np.ndarray:
        if text in self._encode_cache:
            return self._encode_cache[text]
        if not text or not text.strip():
            return np.zeros(self.dim)

        text_l = self._normalize_text(text)
        words  = [w for w in text_l.split() if w not in STOPWORDS]
        if not words:
            words = text_l.split()

        v = np.zeros(self.dim)

        concept_hits = 0
        for word in words:
            c_vec = self._word_to_concept(word)
            if c_vec is not None:
                v += c_vec * (2.0 / max(len(words), 1))
                concept_hits += 1

        for word in words:
            tris = _trigrams(word)
            for tri in tris:
                if tri in self._tri_vecs:
                    v += self._tri_vecs[tri] * (0.3 / max(len(tris), 1) / max(len(words), 1))

        for word in words:
            if self._word_to_concept(word) is None:
                h = 0
                for ch in word:
                    h = (h * 31 + ord(ch)) & 0xFFFFFFFF
                _bucket_start = self.dim // 8
                _bucket_range = self.dim // 8
                idx = _bucket_start + (h % _bucket_range)
                if idx < self.dim:
                    v[idx] += 0.5 / max(len(words), 1)

        tr_chars = set("çğışöüÇĞİŞÖÜ")
        tr_count = sum(1 for c in text_l if c in tr_chars)

        norm = np.linalg.norm(v)
        v = v / (norm + 1e-8) if norm > 0 else v

        # Metadata slots set after normalization so exact values are preserved
        _m = self.dim - 64   # metadata region starts at dim-64
        v[_m + 0] = min(len(text_l) / 100.0, 1.0)
        v[_m + 1] = min(len(words)  / 20.0,  1.0)
        v[_m + 2] = min(concept_hits / max(len(words), 1), 1.0)
        v[_m + 3] = 1.0 if "?" in text else 0.0
        v[_m + 4] = 1.0 if "!" in text else 0.0
        v[_m + 5] = tr_count / max(len(text_l), 1)

        self._encode_cache[text] = v
        return v

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """N texts → (N, dim) float32. Reuses per-text cache."""
        vecs = [self.encode(t) for t in texts]
        if not vecs:
            return np.empty((0, self.dim), dtype=np.float32)
        try:
            from mda.core.accelerator import HAS_TORCH, to_t
            if HAS_TORCH:
                import torch
                return torch.stack([to_t(v) for v in vecs]).cpu().numpy()
        except ImportError:
            pass
        return np.stack(vecs).astype(np.float32)

    def register_concept(self, surface: str, category: str = "") -> np.ndarray:
        """
        Register an entity surface as a named concept in the encoder.
        Called during load() for every entity so encode() can find it.

        Uses a deterministic seed so the same surface always gets the
        same vector — reproducible across runs and independent of load order.
        Seed space starts at 100_000 to avoid collision with CONCEPT_SEEDS
        (which max out at 911).
        """
        from mda.core.bind import random_vector
        key = surface.upper().replace(" ", "_")

        if key in self._concepts:
            return self._concepts[key]

        seed = 0
        for i, ch in enumerate(surface.lower()):
            seed = (seed * 31 + ord(ch)) & 0xFFFFFFFF
        seed = (seed % (2**31 - 100_000)) + 100_000

        vec = normalize(random_vector(self.dim, seed=seed))
        self._concepts[key] = vec

        # Register word mappings so encode() hits this concept during text processing
        for word in surface.lower().split():
            if word not in EN_CONCEPT_MAP and len(word) > 3:
                EN_CONCEPT_MAP[word] = key

        # Register trigrams for partial-match encoding
        for word in surface.lower().split():
            for tri in _trigrams(word):
                if tri not in self._tri_vecs:
                    tri_seed = sum(ord(c) * (i + 1) for i, c in enumerate(tri))
                    self._tri_vecs[tri] = normalize(
                        np.random.default_rng(tri_seed % (2**31)).normal(0, 1, self.dim)
                    )

        return vec

    def similarity(self, a: str, b: str) -> float:
        from mda.core.bind import cosine
        return cosine(self.encode(a), self.encode(b))
