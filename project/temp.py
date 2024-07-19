import itertools


def sum_of_numbers(input_string):
    # Split the input string into lines
    lines = input_string.split("\n")

    # Initialize the sum of numbers
    total_sum = 0

    # Iterate through each line
    for line in lines:
        # Extract numbers from each line
        # Using a generator expression to find numbers in a line
        numbers = [
            float(num) for num in line.split() if num.replace(".", "", 1).isdigit()
        ]
        if len(numbers) == 0:
            continue
        if len(numbers) > 1:
            raise Exception("Each line should contain exactly one number")

        # Sum the numbers in this line and add to the total sum
        total_sum += sum(numbers) - 1

    return total_sum


# Example input string
input_string = """
                      412
          23
          26
          25
"""

with open("file.log", "r") as file:
    # Read the content of the file
    input_string = file.read()

# Calculate the sum of numbers in the string
result = sum_of_numbers(input_string)
print("The sum of the numbers is:", result)
