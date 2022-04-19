package main

import (
	"github.com/faiface/pixel"
	"math"
	"math/rand"
)

const Timestep = 1.0 / 60

type Simulation struct {
	RbtPos  pixel.Vec
	TarPos  pixel.Vec
	Fitness float64
	Steps   int
}

func (s *Simulation) Step(velocity pixel.Vec) {
	if velocity.Len() > 1 {
		velocity = velocity.Unit()
	}
	scaled := velocity.Scaled(Timestep * 80)
	s.RbtPos = s.RbtPos.Add(scaled)
	//f := 0.3 * DropOff01(s.RbtPos.Sub(s.TarPos).Len(), 60)
	//f += 0.7 * DropOff01(pixel.V(400, 400).Sub(s.TarPos).Len(), 60)
	//s.Fitness += f
	dir := s.RbtPos.Sub(s.TarPos)
	if dir.Len() < 27 {
		//dirInvX, dirInvY := dir.XY()
		s.TarPos = s.TarPos.Sub(dir.Unit().Scaled(Timestep * 200))
	}
	//dirCenter := s.TarPos.Sub(pixel.V(400, 400))
	//s.TarPos = s.TarPos.Sub(dirCenter.Unit().Scaled(Timestep * 10))
	s.Steps++
}

func (s *Simulation) GetFitness() float64 {
	f := 0.3 * DropOff01(s.RbtPos.Sub(s.TarPos).Len(), 60)
	f += 0.7 * DropOff01(pixel.V(400, 400).Sub(s.TarPos).Len(), 60)
	return f
	//return s.Fitness / float64(s.Steps)
}

func NewSimulation() *Simulation {
	return &Simulation{
		RbtPos:  pixel.V(50, 50),
		TarPos:  pixel.V(400, 400),
		Fitness: 0,
		Steps:   0,
	}
}

func (s *Simulation) RandomiseStartingPositions() {
	s.TarPos = pixel.V(rand.Float64()-0.5, rand.Float64()-0.5).Unit().Scaled(200).Add(pixel.V(400, 400))
	s.RbtPos = pixel.V(rand.Float64()-0.5, rand.Float64()-0.5).Unit().Scaled(200).Add(s.TarPos)
}

func (s Simulation) GetInputs() []float64 {
	ins := make([]float64, 5)
	delta := s.RbtPos.Sub(s.TarPos)
	deltaScaled := delta.Scaled(1 / 800.0)
	ins[0], ins[1] = deltaScaled.XY()
	ins[0], ins[1] = ssqrt(ins[0]), ssqrt(ins[1])
	delta = s.RbtPos.Sub(pixel.V(400, 400))
	deltaScaled = delta.Scaled(1 / 800.0)
	ins[2], ins[3] = deltaScaled.XY()
	ins[2], ins[3] = ssqrt(ins[2]), ssqrt(ins[3])
	ins[4] = 1
	return ins
}

func DropOff01(dist, dropOff float64) float64 {
	return 1.0 / (dist/dropOff + 1.0)
}

func ssqrt(x float64) float64 {
	return math.Copysign(math.Sqrt(math.Abs(x)), x)
}
