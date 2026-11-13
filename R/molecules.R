#' Molecular graphs from atoms and bonds
#'
#' `molecule_graph()` builds an undirected [graph_data()] object with one
#' node per atom and one edge pair per bond. Node features are a one hot
#' encoding of the element over `elements` (an extra "other" column catches
#' anything else); edge features are a one hot encoding of the bond order
#' over `bond_orders`, repeated for both directions.
#'
#' graphlearnR does not parse SMILES or compute Bemis-Murcko scaffolds; use
#' a chemistry toolkit such as RDKit for that and pass atoms, bonds and
#' scaffold identifiers here and to [split_graphs()].
#'
#' `dataset_synthetic_molecules()` generates small random molecules: a
#' carbon backbone, optionally closed into a ring of 5 or 6 atoms, with
#' random heteroatoms (N, O) and side chains. The label `graph_y` is
#' `"ring"` or `"chain"`, and the list's `"scaffold"` attribute holds a
#' synthetic scaffold identifier per molecule (ring size and heteroatom
#' count in the ring) so that scaffold splits can be demonstrated. The data are synthetic and carry no
#' chemical meaning.
#'
#' @param atoms Character vector of element symbols.
#' @param bonds Integer matrix or data frame with columns `from`, `to` (1
#'   based atom positions) and optionally `order` (bond order, default 1).
#' @param elements Elements given their own feature column.
#' @param bond_orders Bond orders given their own feature column (1.5 for
#'   aromatic).
#' @param graph_y Optional graph label.
#' @return `molecule_graph()`: a `graph_data` object.
#'   `dataset_synthetic_molecules()`: a list of them with a `"scaffold"`
#'   attribute.
#' @examples
#' # ethanol, heavy atoms only
#' g <- molecule_graph(c("C", "C", "O"), rbind(c(1, 2), c(2, 3)))
#' g
#' mols <- dataset_synthetic_molecules(20, seed = 1)
#' table(vapply(mols, function(m) as.character(m$graph_y), ""))
#' @export
molecule_graph <- function(atoms, bonds, elements = c("C", "N", "O", "S", "F", "Cl", "Br", "P", "I"),
                           bond_orders = c(1, 2, 3, 1.5), graph_y = NULL) {
  atoms <- as.character(atoms)
  if (!length(atoms)) stopf("`atoms` must not be empty.")
  bonds <- as.data.frame(bonds)
  if (ncol(bonds) < 2L) stopf("`bonds` needs `from` and `to` columns.")
  if (!all(c("from", "to") %in% names(bonds))) {
    # Unnamed input: columns are from, to and (when present) the bond order.
    names(bonds)[1:2] <- c("from", "to")
    if (ncol(bonds) >= 3L && !"order" %in% names(bonds)) names(bonds)[3L] <- "order"
  }
  if (is.null(bonds$order)) bonds$order <- rep(1, nrow(bonds))
  check_index(bonds$from, "bonds$from")
  check_index(bonds$to, "bonds$to")
  bad <- setdiff(unique(bonds$order), bond_orders)
  if (length(bad)) stopf("Unknown bond order(s): %s.", paste(bad, collapse = ", "))
  x <- outer(atoms, elements, "==") * 1
  x <- cbind(x, other = as.numeric(!atoms %in% elements))
  colnames(x) <- c(elements, "other")
  ea <- outer(bonds$order, bond_orders, "==") * 1
  colnames(ea) <- paste0("bond_", bond_orders)
  g <- graph_data(x = x, edge_index = rbind(as.integer(bonds$from), as.integer(bonds$to)),
                  edge_attr = ea, num_nodes = length(atoms), directed = FALSE)
  g$graph_y <- graph_y
  g
}

#' @rdname molecule_graph
#' @param n Number of molecules.
#' @param seed Optional seed; the caller's random stream is left untouched.
#' @export
dataset_synthetic_molecules <- function(n = 100L, seed = NULL) {
  out <- with_optional_seed(seed, lapply(seq_len(n), function(i) {
    ring <- stats::runif(1) < 0.5
    m <- if (ring) sample(5:6, 1) else sample(4:8, 1)
    atoms <- sample(c("C", "C", "C", "N", "O"), m, replace = TRUE)
    bonds <- cbind(seq_len(m - 1L), seq_len(m)[-1L])
    if (ring) bonds <- rbind(bonds, c(m, 1L))
    order <- ifelse(stats::runif(nrow(bonds)) < 0.2, 2, 1)
    side <- sample(0:2, 1)
    for (k in seq_len(side)) {
      atoms <- c(atoms, sample(c("C", "O", "N"), 1))
      bonds <- rbind(bonds, c(sample.int(m, 1), length(atoms)))
      order <- c(order, 1)
    }
    scaffold <- if (ring) sprintf("ring%d_het%d", m, sum(atoms[seq_len(m)] != "C")) else "acyclic"
    g <- molecule_graph(atoms, data.frame(from = bonds[, 1], to = bonds[, 2], order = order),
                        graph_y = factor(if (ring) "ring" else "chain", levels = c("chain", "ring")))
    list(g = g, scaffold = scaffold)
  }))
  structure(lapply(out, `[[`, "g"), scaffold = vapply(out, `[[`, "", "scaffold"))
}
