from mda import MDA

memory = MDA()

# Teach facts
memory.learn("The capital of Veloria is Aranthos.")
memory.learn("Aranthos was founded by Queen Seraphel in 412 AE.")

# Retrieve context — returns memory string ready for LLM injection
context = memory.context_for("Who founded the capital?")
print(context)
# → [MEMORY] Aranthos was founded by Queen Seraphel in 412 AE.