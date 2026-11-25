import { render, screen } from '@testing-library/react'
import App from './App'

it('renders product name', () => {
  render(<App />)
  expect(screen.getByText(/Mustang Workshop Assistant/)).toBeInTheDocument()
})
