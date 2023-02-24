package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/JoshPattman/goevo"
	"github.com/faiface/pixel"
	"github.com/faiface/pixel/imdraw"
	"github.com/faiface/pixel/pixelgl"
	"golang.org/x/image/colornames"
)

var (
	ReplayBest             bool
	StartWithPreTrainedNet bool
	NetName                string
	NetNum                 string
)

const EnableRecurrent = false

func fitness(g *goevo.Genotype) float64 {
	p := goevo.NewPhenotype(g)
	f := 0.0
	// Repeat the simulation a number of times
	numReps := 5
	for rep := 0; rep < numReps; rep++ {
		// Reset environment and agent
		p.ClearRecurrentMemory()
		sim := NewSimulation()
		sim.RandomiseStartingPositions()
		// Repeat for a number of frames
		for i := 0; i < 60*10; i++ {
			outs := p.Forward(sim.GetInputs())
			sim.Step(pixel.V(outs[0], outs[1]))
		}
		f += sim.GetFitness()
	}
	// Average the fitness
	f = f / float64(numReps)
	// Add a penalty for count of hidden nodes
	_, numHid, _ := g.Topology()
	f -= (0.03 * math.Max(float64(numHid)-4, 0))
	return f
}

func main() {
	flag.BoolVar(&ReplayBest, "r", false, "If specified, will load a network and simulate it with a window")
	flag.BoolVar(&StartWithPreTrainedNet, "p", false, "If specified, will load a network from dist as starting network instead of creating a new one")
	flag.StringVar(&NetName, "n", "net", "Specify the name prefixing the stored data abount the networks")
	flag.StringVar(&NetNum, "t", "500", "Specify the net number to load from disk")
	flag.Parse()
	rand.Seed(time.Now().Unix())
	if ReplayBest {
		pixelgl.Run(run)
	} else {
		// Setup initial population
		vis := goevo.NewGenotypeVisualiser()
		count := goevo.NewAtomicCounter()
		pop := make([]*goevo.Genotype, 100)
		numIn := len(Simulation{}.GetInputs())
		orig := goevo.NewGenotype(count, numIn, 2, goevo.ActivationLinear, goevo.ActivationTanh)
		if StartWithPreTrainedNet {
			orig := goevo.NewGenotypeEmpty()
			path := "./nets/" + NetName + "_" + NetNum
			dat, _ := os.ReadFile(path + ".json")
			json.Unmarshal(dat, orig)
		}
		for g := range pop {
			pop[g] = goevo.NewGenotypeCopy(orig)
		}
		// Generational loop
		for g := 1; g <= 50000; g++ {
			bestFitness, bestGeno := math.Inf(-1), goevo.NewGenotypeEmpty()
			for pi := range pop {
				f := fitness(pop[pi])
				if f > bestFitness {
					bestFitness = f
					bestGeno = pop[pi]
				}
			}
			for pi := range pop {
				pop[pi] = goevo.NewGenotypeCopy(bestGeno)
				if rand.Float64() < 0.9 {
					goevo.MutateRandomSynapse(pop[pi], 0.2)
				}
				if rand.Float64() < 0.1 {
					goevo.AddRandomSynapse(count, pop[pi], 0.3, false, 5)
				}
				if rand.Float64() < 0.02 && EnableRecurrent {
					goevo.AddRandomSynapse(count, pop[pi], 0.3, true, 5)
				}
				if rand.Float64() < 0.01 {
					goevo.AddRandomNeuron(count, pop[pi], goevo.ActivationReLU)
				}
			}
			if g%100 == 0 {
				js, _ := json.Marshal(bestGeno)
				name := "nets/" + NetName + "_" + strconv.Itoa(g)
				err := os.WriteFile(name+".json", js, 0644)
				if err != nil {
					panic(err)
				}
				vis.DrawImageToPNGFile(name+".png", bestGeno)
			}
			fmt.Println("Generation: ", g, ", Fitness: ", bestFitness)
		}
	}
}

func run() {
	geno := goevo.NewGenotypeEmpty()
	path := "./nets/" + NetName + "_" + NetNum
	dat, _ := os.ReadFile(path + ".json")
	json.Unmarshal(dat, geno)
	pheno := goevo.NewPhenotype(geno)
	cfg := pixelgl.WindowConfig{
		Title:  "Neat",
		Bounds: pixel.R(0, 0, 800, 800),
		VSync:  true,
	}
	win, err := pixelgl.NewWindow(cfg)
	if err != nil {
		panic(err)
	}
	sim := NewSimulation()
	imd := imdraw.New(nil)
	sim.RandomiseStartingPositions()
	pic, _ := loadPicture(path + ".png")
	sprite := pixel.NewSprite(pic, pic.Bounds())

	for !win.Closed() {
		win.Clear(colornames.Black)
		sprite.Draw(win, pixel.IM.Scaled(pixel.V(0, 0), 0.4).Moved(pixel.V(625, 150)))
		outs := pheno.Forward(sim.GetInputs())
		sim.Step(pixel.V(outs[0], outs[1]))
		// Centre
		imd.Clear()
		imd.Color = pixel.RGB(0, 1, 0)
		imd.Push(pixel.V(400, 400))
		imd.Circle(30, 10)
		imd.Draw(win)
		// Robot cirlce
		imd.Clear()
		imd.Color = pixel.RGB(1, 0, 0)
		imd.Push(sim.RbtPos)
		imd.Circle(5, 10)
		imd.Draw(win)
		// Target circle
		imd.Clear()
		imd.Color = pixel.RGB(0, 0, 1)
		imd.Push(sim.TarPos)
		imd.Circle(15, 10)
		imd.Draw(win)
		win.Update()
	}
}
func loadPicture(path string) (pixel.Picture, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	img, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}
	return pixel.PictureDataFromImage(img), nil
}
