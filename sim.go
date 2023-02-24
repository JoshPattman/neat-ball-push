package main

import (
	"math"
	"math/rand"

	"github.com/faiface/pixel"
)

const Timestep = 1.0 / 60

type Simulation struct {
	RbtPos         pixel.Vec
	BallPos        pixel.Vec
	TargetPos      pixel.Vec
	Fitness        float64
	Steps          int
	HasTouchedBall bool
	BallInCenter   bool
}

func NewSimulation() *Simulation {
	return &Simulation{
		RbtPos:    pixel.V(50, 50),
		BallPos:   pixel.V(400, 400),
		TargetPos: pixel.V(400, 400),
		Fitness:   0,
		Steps:     0,
	}
}

func (s *Simulation) Step(velocity pixel.Vec) {
	if velocity.Len() > 1 {
		velocity = velocity.Unit()
	}
	scaled := velocity.Scaled(Timestep * 80)
	s.RbtPos = s.RbtPos.Add(scaled)
	dir := s.RbtPos.Sub(s.BallPos)
	if dir.Len() < 27 {
		s.BallPos = s.BallPos.Sub(dir.Unit().Scaled(Timestep * 200))
		s.HasTouchedBall = true
	}

	s.BallInCenter = s.TargetPos.Sub(s.BallPos).Len() < 20
	s.Steps++
}

func (s *Simulation) GetFitness() float64 {
	ball2Target := s.TargetPos.Sub(s.BallPos).Len()
	rbt2Ball := s.RbtPos.Sub(s.BallPos).Len()
	// Steps: touch ball, touch ball to center, get far from ball
	if !s.HasTouchedBall {
		return dropOff(rbt2Ball, 1)
	} else if !s.BallInCenter {
		return dropOff(ball2Target, 1) + 1
	} else {
		return (1 - dropOff(rbt2Ball, 1)) + 2
	}
}

func (s *Simulation) RandomiseStartingPositions() {
	s.BallPos = pixel.V(rand.Float64()-0.5, rand.Float64()-0.5).Unit().Scaled(200).Add(s.TargetPos)
	s.RbtPos = pixel.V(rand.Float64()-0.5, rand.Float64()-0.5).Unit().Scaled(200).Add(s.BallPos)
}

func (s Simulation) GetInputs() []float64 {
	ins := make([]float64, 6)
	delta := s.RbtPos.Sub(s.BallPos)
	deltaScaled := delta.Scaled(1 / 800.0)
	ins[0], ins[1] = deltaScaled.XY()
	ins[0], ins[1] = ssqrt(ins[0]), ssqrt(ins[1])
	delta = s.RbtPos.Sub(s.TargetPos)
	deltaScaled = delta.Scaled(1 / 800.0)
	ins[2], ins[3] = deltaScaled.XY()
	ins[2], ins[3] = ssqrt(ins[2]), ssqrt(ins[3])
	ins[4] = 0
	if s.BallInCenter {
		ins[4] = 1
	}
	ins[5] = 1
	return ins
}

// A function that has output between 0 and 1, where `dist=0` returns 1 and `dist=+inf` returns 0
func dropOff(dist, dropOff float64) float64 {
	return 1.0 / (dist/dropOff + 1.0)
}

// Signed sqare root
func ssqrt(x float64) float64 {
	return math.Copysign(math.Sqrt(math.Abs(x)), x)
}
