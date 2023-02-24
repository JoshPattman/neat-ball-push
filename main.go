package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/JoshPattman/goevo"
	"github.com/faiface/pixel"
	"github.com/faiface/pixel/imdraw"
	"github.com/faiface/pixel/pixelgl"
	"golang.org/x/image/colornames"
)

var (
	ReplayBest bool
	NetName    string
	// If enabled, reccurent connections will be allowed
	EnableRecurrent = true
)

// The fitness function. Evaluates a genotype to give it a float64 fitness
func fitness(g *goevo.Genotype) float64 {
	// Create the phenotype
	p := goevo.NewPhenotype(g)
	f := 0.0
	worstF := 100.0
	// Repeat the simulation a number of times for stability
	numReps := 10
	for rep := 0; rep < numReps; rep++ {
		// Reset environment and agent
		p.ClearRecurrentMemory()
		sim := NewSimulation()
		sim.RandomiseStartingPositions()
		// Repeat for a number of frames
		for i := 0; i < 60*10; i++ {
			// Provide the agent with the state and then step the environmnet with the action
			outs := p.Forward(sim.GetInputs())
			sim.Step(pixel.V(outs[0], outs[1]))
		}
		// Add the fitness of this run to the total fitness
		f += sim.GetFitness()
		sf := sim.GetFitness()
		if sf < worstF {
			worstF = sf
		}
	}
	// Average the fitness
	f = f / float64(numReps)

	// Add a penalty for count of synapses
	numSyn := len(g.Synapses)
	f -= (float64(numSyn) * 0.00025)
	return f
}

func main() {
	// Parse command line args
	flag.BoolVar(&ReplayBest, "s", false, "If specified, run the simulation from a trained network instead of training one")
	flag.StringVar(&NetName, "n", "net", "Specify the name of the network to train")
	flag.Parse()
	fmt.Println("Running the loop. Ctrl+C to stop training at any time")
	rand.Seed(time.Now().Unix())
	// Check if this is training or replaying
	if ReplayBest {
		pixelgl.Run(replay)
	} else {
		train()
	}
}

// Load a network from disk and run it in a visual environment. Press R to reset environment
func replay() {
	// Setup the window
	cfg := pixelgl.WindowConfig{
		Title:  "Neat",
		Bounds: pixel.R(0, 0, 800, 800),
		VSync:  true,
	}
	win, err := pixelgl.NewWindow(cfg)
	if err != nil {
		panic(err)
	}

	// Load the genotype from disk
	geno := goevo.NewGenotypeEmpty()
	dat, _ := os.ReadFile("./nets/" + NetName + ".json")
	json.Unmarshal(dat, geno)

	// Create the phenotype
	pheno := goevo.NewPhenotype(geno)

	// Setup the first simulation
	sim := NewSimulation()
	sim.RandomiseStartingPositions()

	// Setup the network visual and drawing
	vis := goevo.NewGenotypeVisualiser()
	pic := pixel.PictureDataFromImage(vis.DrawImage(geno))
	sprite := pixel.NewSprite(pic, pic.Bounds())
	imd := imdraw.New(nil)

	// Until the window is closed
	for !win.Closed() {
		// Clear the window and draw the network visual
		win.Clear(colornames.Black)
		sprite.Draw(win, pixel.IM.Scaled(pixel.V(0, 0), 0.3).Moved(pixel.V(150, 650)))

		// Step the simulation with the networks chosen action
		outs := pheno.Forward(sim.GetInputs())
		sim.Step(pixel.V(outs[0], outs[1]))

		// Draw the center target
		imd.Clear()
		if sim.BallInCenter {
			imd.Color = pixel.RGB(0, 1, 0)
		} else {
			imd.Color = pixel.RGB(1, 1, 0)
		}
		imd.Push(sim.TargetPos)
		imd.Circle(30, 10)
		imd.Draw(win)

		// Draw the agent
		imd.Clear()
		imd.Color = pixel.RGB(1, 0, 0)
		imd.Push(sim.RbtPos)
		imd.Circle(5, 10)
		imd.Draw(win)

		// Draw the ball
		imd.Clear()
		if sim.HasTouchedBall {
			imd.Color = pixel.RGB(1, 0, 1)
		} else {
			imd.Color = pixel.RGB(0, 0, 1)
		}
		imd.Push(sim.BallPos)
		imd.Circle(15, 10)
		imd.Draw(win)

		//Update the window
		win.Update()

		// If we press R, reset the sim and agent
		if win.JustPressed(pixelgl.KeyR) {
			sim = NewSimulation()
			sim.RandomiseStartingPositions()
			pheno.ClearRecurrentMemory()
		}
	}

}

// Create a new empty network as a common ancestor then evolve a new solution
func train() {
	// Setup goevo stuff
	vis := goevo.NewGenotypeVisualiser()
	count := goevo.NewAtomicCounter()

	// Setup initial population
	pop := make([]*goevo.Genotype, 100)
	numIn := len(Simulation{}.GetInputs())
	orig := goevo.NewGenotype(count, numIn, 2, goevo.ActivationLinear, goevo.ActivationTanh)
	for g := range pop {
		pop[g] = goevo.NewGenotypeCopy(orig)
	}

	// Generational loop
	for g := 1; g <= 50000; g++ {
		// Calculate the fitnesses for all agents, also find the best fitness whilst we are doing this
		bestFitness, bestGeno := math.Inf(-1), goevo.NewGenotypeEmpty()
		avgFitness := 0.0
		fitnesses := make([]float64, len(pop))
		for pi := range pop {
			f := fitness(pop[pi])
			if f > bestFitness {
				bestFitness = f
				bestGeno = pop[pi]
			}
			fitnesses[pi] = f
			avgFitness += f
		}
		avgFitness = avgFitness / float64(len(pop))
		// Create a second population of clones drawn from the first population, with bias towards the best individuals
		SortPopulation(pop, fitnesses)
		popNewIndexes := SampleSortedPopulation(pop, len(pop), 0.7, 0.5)
		popNew := make([]*goevo.Genotype, len(pop))
		// For every agent in the new population, possibly mutate it
		for pi := range pop {
			agent := goevo.NewGenotypeCopy(pop[popNewIndexes[pi]])
			if rand.Float64() < 0.9 {
				goevo.MutateRandomSynapse(agent, 0.2)
			}
			if rand.Float64() < 0.1 {
				goevo.AddRandomSynapse(count, agent, 0.3, false, 5)
			}
			if rand.Float64() < 0.02 && EnableRecurrent {
				goevo.AddRandomSynapse(count, agent, 0.3, true, 5)
			}
			if rand.Float64() < 0.01 {
				goevo.AddRandomNeuron(count, agent, goevo.ActivationReLU)
			}
			if rand.Float64() < 0.01 {
				goevo.PruneRandomSynapse(agent)
			}
			popNew[pi] = agent
		}
		// Set the current population to our new population
		pop = popNew

		// Once every 25 generations, save the best performing agent to disk. Also save a picture of its network
		if g%25 == 0 {
			js, _ := json.Marshal(bestGeno)
			name := "nets/" + NetName
			err := os.WriteFile(name+".json", js, 0644)
			if err != nil {
				panic(err)
			}
			vis.DrawImageToPNGFile(name+".png", bestGeno)
		}
		fmt.Printf("Generation: %v, Best Fitness: %.4f, Average Fitness: %.4f\n", g, bestFitness, avgFitness)
	}
}
