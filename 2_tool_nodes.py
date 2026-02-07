from langchain_core.tools import tool


@tool
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


@tool
def subtract_two_numbers(a: int, b: int) -> int:
    """
    Subtract two numbers
    """

    # The cast is necessary as returned tool call arguments don't always conform exactly to schema
    return int(a) - int(b)


@tool
def safe_divide(a: float, b: float) -> str:
    """
    Divide a by b, with safe handling for division by zero.

    Args:
        a: Numerator.
        b: Denominator.

    Returns:
        A division result or an error message.
    """
    if b == 0:
        return "Cannot divide by zero."
    return a / b


@tool
def get_weather(city: str) -> str:
    """
    Return weather for a city.

    Args:
        city: City name.
    """
    city_norm = city.strip().title()
    known = {
        "New York": "sunny, 25°C",
        "London": "cloudy, 19°C",
        "Mumbai": "humid, 30°C",
        "Tokyo": "clear, 22°C",
    }
    desc = known.get(city_norm, "mild, 24°C")
    return f"The weather in {city_norm} is {desc}."


print(add_two_numbers.run({"a": 5, "b": 6}))  # Works if the tool uses kwargs

print(subtract_two_numbers.run({"a": 10, "b": 6}))  # Works if the tool uses kwargs

print(safe_divide.run({"a": 10, "b": 0}))  # Works if the tool uses kwargs

print(get_weather.run({"city": "New York"}))  # Works if the tool uses kwargs

print("-------------------------------------------")
print("Tool name : " + safe_divide.name)
print("Tool description : " + safe_divide.description)
