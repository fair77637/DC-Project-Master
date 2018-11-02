#!/usr/bin/env python
# Copyright (c) 2007-8 Qtrac Ltd. All rights reserved.
# This program or module is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 2 of the License, or
# version 3 of the License, or (at your option) any later version. It is
# provided for educational purposes and is distributed in the hope that
# it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
# the GNU General Public License for more details.


from PyQt4.QtCore import *
from PyQt4.QtXml import *


CODEC = "UTF-8"
NEWPARA = unichr(0x2029)
NEWLINE = unichr(0x2028)
DATEFORMAT = "ddd MMM d, yyyy"


class Scan(object):
    """A Movie object holds the details of a movie.
    
    The data held are the title, year, minutes length, date acquired,
    and notes. If the year is unknown it is set to 1890. If the minutes
    is unknown it is set to 0. The title and notes are held as QStrings,
    and the notes may contain embedded newlines. Both are plain text,
    and can contain any Unicode characters. The title cannot contain
    newlines or tabs, but the notes can contain both. The date acquired
    is held as a QDate.
    """

    def __init__(self, array=None, spacing=None, params=None, notes=None):
        self.array = array
        self.spacing = spacing
        self.params = params # set this to a, b, c, d
        self.notes = notes

    @staticmethod
    def formats():
        return "*.dcm *.nii *.nii.gz *.nrrd"

    def setActions(self, symmetry, midplaneMask, brainMask, haematomaMask, ventricleMask):
        self.symmetry = symmetry
        self.midplaneMask = midplaneMask
        self.brainMask = brainMask
        self.haematomaMask = haematomaMask
        self.ventricleMask = ventricleMask


