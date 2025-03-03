
class NBasisError(Exception): ## if number of basis funcs is out of range
    pass

class DimensionError(Exception): ## if cosmology does not have the correct dimensions
    pass

class KRangeError(Exception): ## if k range is not within cobra range
    pass

class ParamRangeError(Exception): ## if range is not default or extended
    pass

class ConfigError(Exception): ## if arguments are not configured correctly
    pass

