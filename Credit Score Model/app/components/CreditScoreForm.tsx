'use client'

import { useState } from 'react'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

export function CreditScoreForm() {
  const [formData, setFormData] = useState({
    income: '',
    age: '',
    loan_amount: '',
    credit_score: ''
  })
  const [prediction, setPrediction] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      })
      const data = await response.json()
      setPrediction(data.prediction)
    } catch (error) {
      console.error('Error:', error)
    }
    setLoading(false)
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    })
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <Label htmlFor="income">Income</Label>
        <Input
          id="income"
          name="income"
          type="number"
          required
          value={formData.income}
          onChange={handleChange}
        />
      </div>
      <div>
        <Label htmlFor="age">Age</Label>
        <Input
          id="age"
          name="age"
          type="number"
          required
          value={formData.age}
          onChange={handleChange}
        />
      </div>
      <div>
        <Label htmlFor="loan_amount">Loan Amount</Label>
        <Input
          id="loan_amount"
          name="loan_amount"
          type="number"
          required
          value={formData.loan_amount}
          onChange={handleChange}
        />
      </div>
      <div>
        <Label htmlFor="credit_score">Current Credit Score</Label>
        <Input
          id="credit_score"
          name="credit_score"
          type="number"
          required
          value={formData.credit_score}
          onChange={handleChange}
        />
      </div>
      <Button type="submit" disabled={loading}>
        {loading ? 'Predicting...' : 'Predict Credit Score'}
      </Button>
      {prediction !== null && (
        <div className="mt-4">
          <h3 className="text-lg font-semibold">Predicted Credit Score:</h3>
          <p className="text-2xl font-bold">{prediction}</p>
        </div>
      )}
    </form>
  )
}

