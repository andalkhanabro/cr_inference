# import nifty8.re as jft   # numerical ift library
import matplotlib.pyplot as plt 
import nifty8 as ift
import os

def make_parameter_priors():

    domain = ift.DomainTuple.make(ift.UnstructuredDomain(1))

    def truncated_normal(mean, lower, upper, scale=None):
        scale = scale or (upper - lower) / 6
        xi = ift.from_random(domain, 'normal')
        y = mean + xi * scale
        return ift.clip(y, lower, upper)

    def uniform(a, b):
        xi = ift.from_random(domain, 'normal')
        return a + (b - a) * ift.sigmoid(xi)

    parameters = {
        'xmax1': truncated_normal(mean=650, lower=400, upper=700),
        'deltaxmax': truncated_normal(mean=300, lower=200, upper=400),
        'nmax1': uniform(a=1e5, b=3e5),
        'n_fac': truncated_normal(mean=0.5, lower=0.1, upper=0.9)
    }

    return parameters


xmax1_vals = []

for iterations in range(int(1e6)):

    xmax1_vals.append(make_parameter_priors()["xmax1"].val[0])


plt.hist(xmax1_vals, bins=30, color='lightblue', edgecolor='black', alpha=0.7)
plt.axvline(x=400)
plt.axvline(x=700)
plt.xlabel('xmax1')
plt.ylabel('Frequency')
plt.title('Histogram of draws from prior for xmax1')

output_dir = '/cr/users/abro/cr_inference/cr_inference/plots'
os.makedirs(output_dir, exist_ok=True)
full_path = os.path.join(output_dir, f'xmax1_dist.png')
plt.savefig(full_path, dpi=300, bbox_inches='tight', transparent=False)




    



