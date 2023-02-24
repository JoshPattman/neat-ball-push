package main

import (
	"math"
	"math/rand"
	"sort"
)

// Sample `number` of agents from the sorted `population` (index 0 has best fitness).
// `maxPercentile`: anything outside of this percentile will not survive. EG 0.2 means new pop will only be drawn from top 0.2 percent of pop.
// `fitnessBias` (0-1): 0 - all pop within maxPercentile is considered equally. 1 - the best agent is the only one that is picked
func SampleSortedPopulation[T any](population []T, number int, maxPercentile, fitnessBias float64) []int {
	distribution := func(x float64) float64 { return maxPercentile * math.Pow(x, 1/(1-fitnessBias)) }
	newPop := make([]int, number)
	for pi := range newPop {
		percentile := distribution(rand.Float64())
		newPop[pi] = int(percentile * float64(len(population)))
	}
	return newPop
}

type sortPair[T any] struct {
	data    T
	fitness float64
}

// Sorted the `population` and `fitnesses` based on highest fitness first
func SortPopulation[T any](population []T, fitnesses []float64) {
	zipped := make([]sortPair[T], len(population))
	for pi := range population {
		zipped[pi] = sortPair[T]{population[pi], fitnesses[pi]}
	}
	sort.Slice(zipped, func(i, j int) bool { return zipped[i].fitness > zipped[j].fitness })
	for pi := range population {
		population[pi] = zipped[pi].data
		fitnesses[pi] = zipped[pi].fitness
	}
}
