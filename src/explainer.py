import collections
from functools import reduce
from pysat.examples.hitman import Hitman
from pysat.solvers import Solver as SATSolver




##reused code
###Needed to get hypos for mhs_mus_enumeration
#### hypos example [1, -2, 3, 12]: those are ids of thresholds for each feature (for example for a given sample feature 1 is 5; and we have threshlod splitting for feature 1 is 6 with id 1, as 5 < 6 so we consider id 1)
def get_literals(sample_internal,imaps,ivars,feature_names_as_array_strings):
        """
            Translate an instance to a list of propositional literals.
        """
        lits = []
        imaps={str(k):v for k,v in imaps.items()}
        for feat, fval in zip(feature_names_as_array_strings, sample_internal):
            if feat in imaps:
                # determining the right interval and the corresponding variable
                for ub, fvar in zip(imaps[feat], ivars[feat]):
                    if ub == '+' or fval < ub:
                        lits.append(fvar)
                        break

        return lits

#### how vpos is defined
###Needed to get fcats for mhs_mus_enumeration part1
#here vpos {1: 0, 2: 1, 3: 2, 5: 2, 7: 2, 9: 2, 11: 2, 10: 2, 12: 3, 14: 3, 13: 3} means  1 is id for feature '0', 3, 5, 7, 9, 11 ,10 are ids for feature '1' and 12, 13, 14 are ids for feature '2'
def make_varpos(ivars):
        """
            Traverse all the vars and get their positions in the list of inputs.
        """
        vpos, pos = {}, 0

        for feat in ivars:
            if len(ivars[feat]) == 2:
                for lit in ivars[feat]:
                    if abs(lit) not in vpos:
                        vpos[abs(lit)] = pos
                        pos += 1
            else:
                for lit in ivars[feat]:
                    if abs(lit) not in vpos:
                        vpos[abs(lit)] = pos
                pos += 1
        return vpos


###Needed to get fcats for mhs_mus_enumeration part2
def get_feature_categories(ivars,feature_names_as_array_strings):
    """
        Get the categories of features.
    """    
    categories = collections.defaultdict(lambda: [])
    ivars={str(k):v for k,v in ivars.items()}
    # feature_names_as_array_strings=['0', '1', '2', '3']
    vpos = make_varpos(ivars)
    for f in feature_names_as_array_strings:
        if f in ivars:
            if len(ivars[f]) == 2:
                categories[f.split('_')[0]].append(vpos[ivars[f][0]])
            else:
                for v in ivars[f]:
                    # this has to be checked and updated
                    categories[f].append(vpos[abs(v)])
    # these are the result indices of features going together
    fcats = [[min(ftups), max(ftups)] for ftups in categories.values()]
    return fcats

def cats2hypo(scats,hypos,fcats):
    """
        Translate selected categories into propositional hypotheses.
    """

    return list(reduce(lambda x, y: x + hypos[y[0] : y[1] + 1],
        [fcats[c] for c in scats], []))

def predict( sample,self_hypos,fcats,nofcl,formula):
        """
            Run the encoding and determine the corresponding class.
        """
        print("predict using encoding")
        sample=list(sample)
        v2cat = {}
        for i, cat in enumerate(fcats):
            for v in range(cat[0], cat[1] + 1):
                v2cat[self_hypos[v]] = i
        solver = SATSolver(name='g3')
        solver.append_formula(formula[0].formula)
        assert solver.solve(assumptions=self_hypos), 'Formula must be satisfiable!'
        model = solver.get_model()
        # computing all the class scores
        scores = {}
        for clid in range(nofcl):
            # computing the value for the current class label
            scores[clid] = 0

            for lit, wght in formula[clid].leaves:
                             if model[abs(lit) - 1] > 0:
                                 scores[clid] += wght
        # returning the class corresponding to the max score
        return v2cat,max(list(scores.items()), key=lambda t: t[1])[0]


def mhs_mus_enumeration(oracle, allcats, v2cat,hypos, fcats,verbose=0, xnum=1, smallest=True):
    """
    Enumerate subset- and cardinality-minimal explanations.
    """
    # result
    expls = []
    print('''''''''''''''mhs_mus_enumeration''''''''''''''')
    with Hitman(bootstrap_with=[allcats], htype='sorted' if smallest else 'lbx') as hitman:
        # main loop
        iters = 0
        while True:
            hset = hitman.get()
            iters += 1
            if hset is None:
                break
            cats2hypos = cats2hypo(hset,hypos,fcats)
            coex = oracle.get_coex(cats2hypos, early_stop=True)
            if coex:
                to_hit = []
                satisfied, unsatisfied = [], []
                removed = list(set(hypos).difference(set(cats2hypos)))
                for h in removed:
                    if coex[abs(h) - 1] != h:
                        unsatisfied.append(v2cat[h])
                    else:
                        hset.append(v2cat[h])
                unsatisfied = list(set(unsatisfied))
                hset = list(set(hset))
                for h in unsatisfied:
                    if oracle.get_coex(cats2hypo(hset + [h],hypos,fcats), early_stop=True):
                        hset.append(h)
                    else:
                        to_hit.append(h)
                hitman.hit(to_hit)
            else:
                expls.append(hset)

                if len(expls) != xnum:
                    hitman.block(hset)
                else:
                    break
    # print("expls", expls)
    return expls

def GenerateExplanation(out_id,sample_internal,expls,hypos,fcats,v2feat,preamble,target_name):
    result=[]
    for expl in expls:
                    hyps = list(reduce(lambda x, y: x + hypos[y[0]:y[1]+1], [fcats[c] for c in expl], []))
                    expl = sorted(set(map(lambda v:v2feat[v], hyps)))
                    print(expl)
                    preamble = [preamble[i] for i in expl]
                    label = target_name[out_id] #target_name=[0,1,2]; 0,1,2 are the classes
                    print('  explanation: "IF {0} THEN {1}"'.format(' AND '.join(preamble), label))
                    print('  # hypos left:', len(expl))
                    result.append((sample_internal,'  explanation: "IF {0} THEN {1}"'.format(' AND '.join(preamble), label)))
    return  result
    