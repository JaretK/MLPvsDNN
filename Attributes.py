"""
Contains attributes for different model assessment conditions
"""

from abc import ABCMeta, abstractproperty

class AbstractAttribute:
    """Abstract base class for Attributes"""
    __metaclass__ = ABCMeta

    @abstractproperty
    def normalized_attributes(self):
        pass

    @abstractproperty
    def categorical_attributes(self):
        pass

    def get_all_attributes(self):
        l = []
        l.extend(self.normalized_attributes)
        l.extend(self.categorical_attributes)
        return l

class Attributes:
    _TCosts = 'total_costs_inflation_adjusted'
    class _sparcs_attributes(AbstractAttribute):
        normalized_attributes = []
        categorical_attributes = [
            'age_group',
            'apr_risk_of_mortality',
            'apr_severity_of_illness_description',
            'ethnicity',
            'gender',
            'race',
            'type_of_admission',
        ]

if __name__ == '__main__':
    pass
