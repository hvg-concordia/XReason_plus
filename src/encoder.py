from pysat.formula import IDPool, CNF
import collections
import six
from pysat.formula import IDPool, CNF

ClassEnc = collections.namedtuple('ClassEnc', ['formula', 'leaves', 'trees'])

def get_thresholds(trees):
    thresholds={}
    def traverse(node,thresholds):
            if 'threshold' in node:
                if node['split_feature'] not in thresholds:
                    thresholds[node['split_feature']]=[]
                if (node['threshold']not in thresholds[node['split_feature']]):
                      thresholds[node['split_feature']].append(node['threshold'])
            if 'left_child' in node:
                traverse(node['left_child'],thresholds)
            if 'right_child' in node:
                traverse(node['right_child'],thresholds)
    for tree in trees:
        traverse(tree['tree_structure'],thresholds)
    thresholds = dict(sorted(thresholds.items()))
    thresholds = {f: sorted(thresholds[f]) + ['+'] for f in six.iterkeys(thresholds)}
    return thresholds

#reused code
def encode(nofcl,thresholds,idmgr):
    ClassEnc = collections.namedtuple('ClassEnc', ['formula', 'leaves', 'trees'])
    Intervals=thresholds
    vid2fid = {}
    enc = {}
    for j in range(nofcl):
        enc[j] = ClassEnc(formula=CNF(), leaves=[], trees=[])

    # common clauses shared by all classes
    enc['common'] = []

    # path encodings
    enc['paths'] = collections.defaultdict(lambda: [])
    imaps, ivars, lvars = {}, {}, {}
    for feat, intvs in six.iteritems(Intervals):
        imaps[feat] = {intvs[i]: i for i in range(len(intvs))}
        ivars[feat] = [None for i in range(len(intvs))]
        lvars[feat] = [None for i in range(len(intvs))]

        # separate case of the first interval
        
        lvars[feat][0] = idmgr.id('{0}_lvar{1}'.format(feat, 0))
        ivars[feat][0] = lvars[feat][0]
        for i in range(1, len(intvs) - 1):
        
            lvar = idmgr.id('{0}_lvar{1}'.format(feat, i))
            ivar = idmgr.id('{0}_ivar{1}'.format(feat, i))
            prev = lvars[feat][i - 1]
            # order encoding
            enc['common'].append([-prev, lvar])

            # domain encoding
            enc['common'].append([-ivar, -prev])
            enc['common'].append([-ivar,  lvar])
            enc['common'].append([ ivar,  prev, -lvar])

            # saving the variables
            lvars[feat][i] = lvar
            ivars[feat][i] = ivar
        # separate case of the last interval (till "+inf")
        lvars[feat][-1] = -lvars[feat][-2]
        ivars[feat][-1] =  lvars[feat][-1]
        # finally, enforcing that there is at least one interval
        if len(intvs) > 2:
            enc['common'].append(ivars[feat])
    # mapping variable ids to feature idsP
    for feat in ivars:
        for v, ub in zip(ivars[feat], Intervals[feat]):
            vid2fid[v] = (feat, ub)
    return enc,imaps,lvars,ivars,intvs,vid2fid

def get_id(root, thresholds, lvars):
    if isinstance(root, dict):
        f = root['split_feature']
        th = root['threshold']
        pos = thresholds[f].index(th)
        id = lvars[f][pos]
        return id
    else:
        raise TypeError("Expected a dictionary for 'root', but got a list or other type.")



## having dictionary {leaf:[sign*node (- if it is left and + if it iw right)]}: transform it into list of paths to that leaf
def transform(result):
    for key,val in result.items():
        l=[[-x for x in val]+[key]]
        result[key]=[[i,-key]for i in val]+l
    return result

def traverse_tree(idmgr, node, path, result, thresholds, lvars, leaves_weights={}):
    if 'leaf_index' in node:
        leaf_id = idmgr.id()
        result[leaf_id] = path[:]
        leaves_weights[leaf_id] = node['leaf_value']
    else:
        try:
            id_ = get_id(node, thresholds, lvars)
        except TypeError as e:
            print(f"Error in get_id: {e}")
            print(f"Node received: {node}")
            return result, leaves_weights
            
        left_path = path + [id_]
        right_path = path + [-id_]
        
        traverse_tree(idmgr, node['left_child'], left_path, result, thresholds, lvars, leaves_weights)
        traverse_tree(idmgr, node['right_child'], right_path, result, thresholds, lvars, leaves_weights)
    return result, leaves_weights

def swap_literals(enc):
        # Swap positive and negative literals in each clause of the formula
        swapped_formula = enc.formula # Make a copy to avoid modifying the original formula
        for i in range(len(swapped_formula.clauses)):
            clause = swapped_formula.clauses[i]
            swapped_clause = [literal for literal in clause if literal != 0]
            swapped_formula.clauses[i] = swapped_clause
        # Swap the signs in the leaves
        swapped_leaves = [(leaf[0], -leaf[1]) for leaf in enc.leaves]
        # Return the modified ClassEnc object
        return ClassEnc(swapped_formula, swapped_leaves, enc.trees)
