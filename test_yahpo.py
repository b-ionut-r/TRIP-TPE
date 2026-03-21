import sys
from yahpo_gym import BenchmarkSet

b = BenchmarkSet("rbv2_svm")
b.set_instance("4134")
cs = b.get_opt_space()
print(cs)
print("Hyperparameters:")
for hp in cs.get_hyperparameters():
    print(hp.name, type(hp), hasattr(hp, 'choices'), hasattr(hp, 'value'))

cfg = {}
for hp in cs.get_hyperparameters():
    if hasattr(hp, 'choices'):
        cfg[hp.name] = hp.choices[0]
    elif hasattr(hp, 'lower'):
        cfg[hp.name] = hp.lower
    elif hasattr(hp, 'value'):
        cfg[hp.name] = hp.value

import ConfigSpace as CS
try:
    clean_config = CS.Configuration(cs, values=cfg)
    print("Success")
except Exception as e:
    print("Error:", str(e))
