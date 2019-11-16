import cv2


class Player:
    def __init__(self, color=None, settlement_positions=None):
        self.color = color
        self.settlement_positions = settlement_positions

    def update_settlement_positions(self, image):
        '''Using the player's color, look through the image and find the
           positions of the player's settlements'''
        pass
