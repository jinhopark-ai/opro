import string
import hashlib


def remove_punctuation_from_string(input_string, is_filename=True):
  """Remove punctuations from string to comply with filename requirements."""
  # remove punctuations other than "!", "?", "."
  if is_filename:
    punctuation_subset_str = (
        string.punctuation.replace("!", "").replace("?", "").replace(".", "")
    )
    output_string = input_string.translate(
        str.maketrans("", "", punctuation_subset_str)
    )
    # replace punctuations "!", "?", "." with indicating letters
    output_string = (
        output_string.replace("!", "<EXCLAMATION>")
        .replace("?", "<QUESTION>")
        .replace(".", "<PERIOD>")
    )
  else:
    output_string = input_string.translate(
        str.maketrans("", "", string.punctuation)
    )
  return output_string

def instruction_to_filename(instruction, md5_hashing=True):
  """Convert an instruction string to filename."""
  if md5_hashing:
    m = hashlib.md5()
    m.update(instruction.encode("utf-8"))
    filename = m.hexdigest()
  else:
    # remove punctuations and line break, and give a name to the empty string
    filename = instruction.replace("\n", "")
    filename = remove_punctuation_from_string(repr(filename))
    filename = filename if filename else "<NO INSTRUCTION>"
  return filename

def main():
    instruction = "Step by step:"
    print(instruction_to_filename(instruction))


if __name__ == "__main__":
    main()