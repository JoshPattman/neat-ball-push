package main

import (
	"encoding/json"
	"fmt"
	"github.com/JoshPattman/goevo"
	"github.com/faiface/pixel"
	"github.com/faiface/pixel/imdraw"
	"github.com/faiface/pixel/pixelgl"
	"golang.org/x/image/colornames"
	"image"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"
)

// ReplayBest when true, causes a window to pop up and run a previously trained network. When false, it trains new networks
const ReplayBest = true
const NetName = "net_1"
const NetNum = "8000"

const EnableRecurrent = true

var activations = goevo.ActivationInfo{
	InputActivation:  goevo.LinearActivation,
	HiddenActivation: goevo.RelnActivation,
	OutputActivation: goevo.TanhActivation,
}

func main() {
	rand.Seed(time.Now().Unix())
	if ReplayBest {
		pixelgl.Run(run)
	} else {
		// Setup genotypes
		vis := goevo.NewGenotypeVisualiser()
		mut := goevo.GenotypeMutator{
			MaxNewSynapseValue:      0.2,
			MaxSynapseMutationValue: 0.4,
		}
		crs := goevo.GenotypeCrossover{}
		count := goevo.NewAtomicCounter()
		gts := make([]goevo.Genotype, 100)
		numIn := len(Simulation{}.GetInputs())
		orig := goevo.NewGenotypeFast(numIn, 2, count)
		for g := range gts {
			gts[g] = orig.Copy()
			mut.GrowRandomSynapse(gts[g], count)
			mut.GrowRandomNode(gts[g], count)
		}
		pop := goevo.NewPopulation(gts, activations)
		// Train
		for g := 0; g < 200001; g++ {
			for pi := range pop {
				// set pop[pi].fitness
				f := 0.0
				numReps := 5
				for j := 0; j < numReps; j++ {
					pop[pi].Phenotype.ResetRecurrent()
					sim := NewSimulation()
					sim.RandomiseStartingPositions()
					for i := 0; i < 60*10; i++ {
						outs := pop[pi].Phenotype.Calculate(sim.GetInputs())
						sim.Step(pixel.V(outs[0], outs[1]))
					}
					f += sim.GetFitness()
				}
				f = f / float64(numReps)
				_, numHid, _ := pop[pi].Genotype.GetNumNodes()
				pop[pi].Fitness = f - (0.04 * math.Max(float64(numHid)-4, 0))
			}
			pop.Repopulate(0.25, func(g1, g2 goevo.Genotype) goevo.Genotype {
				gn := crs.CrossoverSimple(g1, g2)
				if rand.Float64() < 0.9 {
					mut.MutateRandomConnection(gn)
				}
				if rand.Float64() < 0.02 {
					mut.GrowRandomSynapse(gn, count)
				}
				if rand.Float64() < 0.02 && EnableRecurrent {
					mut.GrowRandomRecurrentSynapse(gn, count)
				}
				if rand.Float64() < 0.01 {
					mut.GrowRandomNode(gn, count)
				}
				return gn
			}, activations)
			if g%500 == 0 && g != 0 {
				js, _ := json.Marshal(pop[0].Genotype)
				name := "nets/" + NetName + "_" + strconv.Itoa(g)
				err := os.WriteFile(name+".json", js, 0644)
				if err != nil {
					panic(err)
				}
				vis.DrawImageToPNGFile(name+".png", pop[0].Genotype)
			}
			fmt.Println("Generation: ", g, ", Fitness: ", pop[0].Fitness)
		}
	}
}

func run() {
	geno := &goevo.GenotypeFast{}
	path := "./nets/" + NetName + "_" + NetNum
	dat, _ := os.ReadFile(path + ".json")
	json.Unmarshal(dat, geno)
	pheno := goevo.GrowPhenotype(geno, activations)
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
		outs := pheno.Calculate(sim.GetInputs())
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
