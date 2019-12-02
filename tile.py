import numpy as np


class Tile:
    def __init__(self, corner_points, tile_ortho, resource, settlements,
                 cities, tile_number):
        self.resource = resource
        self.corner_points = corner_points
        self.settlements = settlements
        self.cities = cities
        self.tile_number = tile_number
        self.tile_ortho = tile_ortho

    def disperse(self, num):
        '''Go through cities and settlements and disperse this resource
        to them
        '''
        to_disperse = []
        if num == self.tile_number:
            for s in self.settlements:
                to_disperse.append((s, self.resource))
        return to_disperse
