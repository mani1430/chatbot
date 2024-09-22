import language_tool_python

class Grammer:
    def __init__(self):
        self.tool = language_tool_python.LanguageTool('en-US')

    def correct_grammer(self, sentence):
        matches = self.tool.check(sentence)
        corrected_text = language_tool_python.utils.correct(sentence, matches)
        return corrected_text