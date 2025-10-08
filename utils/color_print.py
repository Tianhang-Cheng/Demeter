import termcolor

# error message
def red_print(text):
    print(termcolor.colored(text, 'red'))
# hint message
def blue_print(text):
    print(termcolor.colored(text, 'blue'))
# success message
def green_print(text):
    print(termcolor.colored(text, 'green'))
def pink_print(text):
    print(termcolor.colored(text, 'magenta'))