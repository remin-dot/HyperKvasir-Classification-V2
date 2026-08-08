"""Shared code for the HyperKvasir multi-model benchmark.

Four models are compared. If each of them implemented its own metrics, timing or
data loading, the numbers would drift apart and the comparison would be
meaningless. So all of that lives here exactly once, and the per-model scripts
under models/ are thin wrappers around it.
"""
