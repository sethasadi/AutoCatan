#! /usr/bin/env python
# File for working with the hexagons of the catan board

class Hexagon:
    """Hexagon class which knows it's position on the grid, and can tell you
    the location of each of it's vertices in cubic or 2D. Also knows it's
    resource and resource number. Knows also how many settlements and cities
    of each color are touching it.
    """

    def __init__(pos, resource="desert", number=8):
        """Initialize the Hexagon with the desired resource and number
        """

        self.position = pos
        self.resource = resource
        self.number = number
