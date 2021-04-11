import heapq


class ScoredArchitecture(object):
    """
    Class for storing an architecture and its score.
    Enables comparison for heapq.
    """
    def __init__(self, model, score, details=None):
        self.model = model
        self.score = score
        self.details = details

    def __lt__(self, other):
        return self.score < other.score


class HallOfFame(object):
    """
    Hall Of Fame of child models.
    Implemented as a fixed size Minimum heap.
    Smallest element is always at the first position:
    self.hof[0], according to heapq documentation.
    """
    def __init__(self, size):
        self.hof = []
        self.size = size

    def add_scored(self, arch_object):
        # If the HallOfFame is full and model score is
        # smaller than HoF's first element: nothing to do
        if len(self.hof) == self.size and arch_object < self.hof[0]:
            return
        heapq.heappush(self.hof, arch_object)
        # If HoF exceeds size: dump smallest element
        if len(self.hof) > self.size:
            heapq.heappop(self.hof)

    def add(self, model, score, details=None):
        arch_object = ScoredArchitecture(model,
                                         score,
                                         details)
        # If the HallOfFame is full and model score is
        # smaller than HoF's first element: nothing to do
        if len(self.hof) == self.size and arch_object < self.hof[0]:
            return
        heapq.heappush(self.hof, arch_object)
        # If HoF exceeds size: dump smallest element
        if len(self.hof) > self.size:
            heapq.heappop(self.hof)

    def get_elements(self):
        # Return list of heap elements in order
        # Smallest -> Greatest
        hof = "arch;score;details\n"
        for i in range(len(self.hof)):
            # Convert class object to a parseable string format
            arch_object = heapq.heappop(self.hof)
            hof += ";".join([str(arch_object.model),
                             str(arch_object.score),
                             str(arch_object.details)])
            hof += '\n'
        return hof
