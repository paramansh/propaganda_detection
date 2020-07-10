from src import predict

class color:
  PURPLE = '\033[95m'
  CYAN = '\033[96m'
  DARKCYAN = '\033[36m'
  BLUE = '\033[94m'
  GREEN = '\033[92m'
  YELLOW = '\033[93m'
  RED = '\033[91m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'
  END = '\033[0m'

input_text = "Just a random text"
with open('input.txt') as f:
  input_text = f.read()

# get_predictions is used to get output span text list
# output = predict.get_predictions(text)
# print(input_text)
# print('Detected span: ')
# print(output)
# print('--' * 50)

output_indices = predict.get_predictions_indices(input_text)

si = 0

for sp in output_indices:
  text = input_text[si : sp[0]]
  print(text, end='')
  text = input_text[sp[0] : sp[1]]
  print(color.BOLD + text + color.END, end='')
  si = sp[1]

text = input_text[si:]
print(text)


