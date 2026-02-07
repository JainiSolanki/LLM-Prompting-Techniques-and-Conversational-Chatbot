def add_two_numbers(a: int, b: int) -> int:
    """
    Add two numbers

    Args:
      a (int): The first number
      b (int): The second number

    Returns:
      int: The sum of the two numbers
    """

    # The cast is necessary as returned tool call arguments don't always conform exactly to schema
    # E.g. this would prevent "what is 30 + 12" to produce '3012' instead of 42
    return int(a) + int(b)


def subtract_two_numbers(a: int, b: int) -> int:
    """
    Subtract two numbers
    """

    # The cast is necessary as returned tool call arguments don't always conform exactly to schema
    return int(a) - int(b)


def safe_divide(a: float, b: float):
    """Divide a by b. Returns None if b == 0."""
    if b == 0:
        return "not possible to divide by 0."
    return a / b


def get_weather(city: str) -> str:
    """Return weather for a given city."""
    city_norm = city.strip().title()
    known = {
        "New York": "sunny, 25°C",
        "London": "cloudy, 19°C",
        "Mumbai": "humid, 30°C",
        "Tokyo": "clear, 22°C",
    }
    desc = known.get(city_norm, "mild, 24°C")
    return f"The weather in {city_norm} is {desc}."


print(add_two_numbers(5, 6))

print(subtract_two_numbers(10, 3))

print(safe_divide(10, 0))

print(get_weather("New York"))
