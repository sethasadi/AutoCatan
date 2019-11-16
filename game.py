import cv2


class Game:
    def __init__(self, player_count):
        self.player_count = player_count
        self.players = []

    def find_player_colors(self):
        '''Look at board and find player colors'''
        pass

    def begin_gameplay(self):
        '''Start the operations of a game like tracking dice rolls and
           player resources'''
        pass

    def end_gameplay(self):
        '''End the gameplay, possibly listing the winner, cleaning up all
           processes'''
        pass
