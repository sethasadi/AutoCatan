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

    def disperse(self):
        '''Go through cities and settlements and disperse this resource
        to them
        '''
