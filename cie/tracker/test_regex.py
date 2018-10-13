import re

integer_pattern = r"-?\d+"

sentence = "I want adjust_value to be -30"


matches = re.findall(integer_pattern, sentence)

print(matches)
