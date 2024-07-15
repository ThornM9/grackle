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

        if numbers[0] == 2:
            total_sum += 1
            continue

        # Sum the numbers in this line and add to the total sum
        total_sum += sum(numbers)

    return total_sum


# Example input string
input_string = """
             594
           4
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           2
           
"""

# Calculate the sum of numbers in the string
result = sum_of_numbers(input_string)
print("The sum of the numbers is:", result)
