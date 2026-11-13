# Heterogeneous graphs

``` r

library(graphlearnR)
```

Many graphs have more than one kind of node and edge: authors write
papers, drugs target proteins, users rate items. A heterogeneous graph
keeps each node type with its own feature space and each relation with
its own edges, so a model can learn relation specific transformations.

## Building one

Node types are given as a named list; each entry is a feature matrix, a
list with `x`, `y`, and masks, or just a node count for types without
features. Relations are keyed `"source__relation__target"`. Indices are
1-based *within each type*, so `author__writes__paper` edge `(2, 3)`
means author 2 wrote paper 3.

``` r

hg <- hetero_graph_data(
  nodes = list(
    author = list(x = matrix(rnorm(8), 4, 2), y = factor(c("ml", "db", "ml", "db"))),
    paper = matrix(rnorm(15), 5, 3),
    venue = 2
  ),
  edges = list(
    author__writes__paper = rbind(c(1, 1, 2, 3, 4), c(1, 2, 3, 4, 5)),
    paper__cites__paper = rbind(c(2, 3, 5), c(1, 1, 4)),
    paper__published_in__venue = rbind(1:5, c(1, 1, 2, 2, 2))
  )
)
hg
#> <hetero_graph_data> 3 node types, 3 relations
#>   node author              4 nodes, 2 features, labeled
#>   node paper               5 nodes, 3 features
#>   node venue               2 nodes
#>   edge author__writes__paper: 5
#>   edge paper__cites__paper: 3
#>   edge paper__published_in__venue: 5
edge_types(hg)
#>                          key    src          rel   dst num_edges reverse
#> 1      author__writes__paper author       writes paper         5    <NA>
#> 2        paper__cites__paper  paper        cites paper         3    <NA>
#> 3 paper__published_in__venue  paper published_in venue         5    <NA>
```

Indices are checked against the size of their own node type, and the
error names the relation and the side.

``` r

try(hetero_graph_data(nodes = list(a = diag(2), b = diag(3)),
                      edges = list(a__r__b = rbind(3, 1))))
#> Error : Invalid hetero_graph_data:
#> * Relation `a__r__b`: source indices must be in 1:2 (type `a`).
```

### Reverse relations

Messages flow along edge direction. Without a reverse of
`author__writes__paper`, papers would never send information to authors.
[`add_reverse_relations()`](https://adamrsophie.github.io/graphlearnR/reference/add_reverse_relations.md)
adds `paper__rev_writes__author` and records the pairing, which the link
splitter uses to prevent leakage.

``` r

hg <- add_reverse_relations(hg)
edge_types(hg)[, c("key", "reverse")]
#>                              key                        reverse
#> 1          author__writes__paper      paper__rev_writes__author
#> 2            paper__cites__paper        paper__rev_cites__paper
#> 3     paper__published_in__venue venue__rev_published_in__paper
#> 4      paper__rev_writes__author          author__writes__paper
#> 5        paper__rev_cites__paper            paper__cites__paper
#> 6 venue__rev_published_in__paper     paper__published_in__venue
```

### Homogeneous view

[`to_homogeneous()`](https://adamrsophie.github.io/graphlearnR/reference/to_homogeneous.md)
places every type in one index space with offsets, pads features to a
common width, and records the original type of each node and edge. This
is useful for exporting to tools that do not know about types.

``` r

flat <- to_homogeneous(hg, target_type = "author")
flat
#> <graph_data> directed graph: 11 nodes, 26 edges
#>   node features [11 x 3]
#>   node labels: factor with 2 levels
flat$metadata$type_ptr
#> author  paper  venue 
#>      0      4      9
table(flat$edge_type)
#> 
#>          author__writes__paper            paper__cites__paper 
#>                              5                              3 
#>     paper__published_in__venue      paper__rev_writes__author 
#>                              5                              5 
#>        paper__rev_cites__paper venue__rev_published_in__paper 
#>                              3                              5
```

## A worked example: drug classes

[`dataset_synthetic_hetero()`](https://adamrsophie.github.io/graphlearnR/reference/dataset_synthetic_hetero.md)
simulates drugs that target proteins and proteins that interact. A
drug’s class is determined by the proteins it targets more than by its
own features, so a model must pass messages across the relation.

``` r

dh <- add_reverse_relations(dataset_synthetic_hetero(seed = 1))
dh
#> <hetero_graph_data> 2 node types, 4 relations
#>   node drug               60 nodes, 4 features, labeled
#>   node protein            80 nodes, 6 features
#>   edge drug__targets__protein: 300
#>   edge protein__interacts__protein: 234
#>   edge protein__rev_targets__drug: 300
#>   edge protein__rev_interacts__protein: 234
```

[`hetero_gnn_model()`](https://adamrsophie.github.io/graphlearnR/reference/hetero_gnn_model.md)
builds one convolution per relation and per layer and sums (or averages)
the results that arrive at each node type. Node types without features
get a learned embedding.

``` r

library(torch)
results <- list()
for (cv in c("sage", "graph_conv", "gat", "rgcn", "hgt")) {
  torch_manual_seed(1)
  m <- hetero_gnn_model(dh, hidden_dim = 32, output_dim = 3, target_type = "drug",
                        conv = cv, dropout = 0.2)
  fit <- train_graph_model(m, dh, "node_classification", target_type = "drug",
                           epochs = 80, lr = 0.01, seed = 1)
  ev <- evaluate_graph_model(fit, dh)
  results[[cv]] <- ev$value[ev$metric == "accuracy"]
}
unlist(results)
#>       sage graph_conv        gat       rgcn        hgt 
#>       1.00       1.00       0.75       1.00       0.75
```

`conv = "rgcn"` shares bases across relations when `num_bases` is set,
which helps when there are many relations with few edges each.

`conv = "hgt"` uses heterogeneous graph transformer layers (Hu et al.
2020), see
[`hgt_conv()`](https://adamrsophie.github.io/graphlearnR/reference/hgt_conv.md).
Keys, queries and values are projected with weights per node type, each
relation has its own attention and message matrices, and the attention
softmax at a node runs over all its incoming edges across relations, so
relations compete for attention.

``` r

layer <- hgt_conv(dh$node_types, names(dh$edges), 8, 8, heads = 2)
ht <- hetero_tensors(dh)
x <- list(drug = torch_randn(60, 8), protein = torch_randn(80, 8))
out <- layer(x, ht$edge_index)
lapply(out, dim)
#> $drug
#> [1] 60  8
#> 
#> $protein
#> [1] 80  8
a <- layer$last_attention$protein
summary(as.numeric(scatter_sum(a$alpha, a$index, 80)$sum(dim = 2)))
#>    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#>       2       2       2       2       2       2
```

The last line sums the attention each protein gives to its neighbors per
head; every protein with incoming edges gets exactly one per head, which
is 2 here.

HGT also has the paper’s relative temporal encoding. Give each relation
an `edge_time`, the time of the edge relative to its target (for example
the citing paper’s year minus the cited paper’s), and set
`temporal_encoding = TRUE`. A sinusoidal encoding of that time
difference is projected and added to the source features before the keys
and values are computed, so the same neighbor can count differently
depending on how long ago the interaction was. Reverse relations get the
negated times.

``` r

dt <- dataset_synthetic_hetero(seed = 1)
set.seed(1)
dt$edges$drug__targets__protein$edge_time <- runif(ncol(dt$edges$drug__targets__protein$edge_index), 0, 5)
dt <- add_reverse_relations(dt)
summary(dt$edges$protein__rev_targets__drug$edge_time)
#>     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
#> -4.96342 -3.61857 -2.38733 -2.45056 -1.28959 -0.06539
torch_manual_seed(1)
m_rte <- hetero_gnn_model(dt, hidden_dim = 32, output_dim = 3, target_type = "drug",
                          conv = "hgt", temporal_encoding = TRUE, dropout = 0.2)
fit_rte <- train_graph_model(m_rte, dt, "node_classification", target_type = "drug",
                             epochs = 80, lr = 0.01, seed = 1)
evaluate_graph_model(fit_rte, dt)[1, ]
#>     metric value
#> 1 accuracy  0.75
```

Relations without `edge_time` use a time difference of zero.

## Relation specific link prediction

[`split_edges_hetero()`](https://adamrsophie.github.io/graphlearnR/reference/split_edges_hetero.md)
splits one relation. Held out edges are removed both from that relation
and from its reverse, otherwise the model could read a test edge
backwards during message passing. Negatives are drawn with the source
from the source type and the target from the target type.

``` r

sp <- split_edges_hetero(dh, "drug__targets__protein", val = 0.1, test = 0.2, seed = 2)
sp
#> <hetero_link_split> drug__targets__protein: train 210, val 30 (+30 neg), test 60 (+60 neg)
held <- paste(sp$test_pos[1, ], sp$test_pos[2, ])
rev <- sp$train_graph$edges$protein__rev_targets__drug$edge_index
any(paste(rev[2, ], rev[1, ]) %in% held)
#> [1] FALSE
```

``` r

torch_manual_seed(2)
m <- hetero_gnn_model(dh, hidden_dim = 32, conv = "sage", dropout = 0)
fit_l <- train_graph_model(m, sp, "link_prediction", epochs = 60, seed = 2)
evaluate_graph_model(fit_l)
#>              metric      value
#> 1           roc_auc 0.53750000
#> 2 average_precision 0.55200494
#> 3               mrr 0.08820665
#> 4            hits@1 0.00000000
#> 5           hits@10 0.21666667
#> 6           hits@50 0.93333333
```

## Typed neighbor sampling

[`sample_hetero_neighbors()`](https://adamrsophie.github.io/graphlearnR/reference/sample_hetero_neighbors.md)
samples incoming edges of every relation hop by hop, starting from seeds
of one or more types. Fanouts can be one number per hop for all
relations, or a named list per relation; a relation left out of the list
is not sampled.

``` r

sg <- sample_hetero_neighbors(dh, seeds = list(drug = c(1, 2)), fanouts = c(3, 2),
                              seed = 1)
lengths(sg$n_id)
#>    drug protein 
#>      10      15
vapply(sg$edges, function(e) ncol(e$edge_index), integer(1))
#>          drug__targets__protein     protein__interacts__protein 
#>                              12                              10 
#>      protein__rev_targets__drug protein__rev_interacts__protein 
#>                               6                              10
```

Local indices map back through `sg$n_id[[type]]`, just as in the
homogeneous sampler.

## Batching

[`hetero_batch()`](https://adamrsophie.github.io/graphlearnR/reference/hetero_batch.md)
combines heterogeneous graphs with the same schema, offsetting indices
separately for every node type and recording a `batch` vector per type.

``` r

hb <- hetero_batch(list(hg, hg))
num_nodes(hb)
#> author  paper  venue 
#>      8     10      4
hb$batch$author
#> [1] 1 1 1 1 2 2 2 2
```
