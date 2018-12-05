/*********************************************************************************************** 
 * Copyright (C) 2018 Dillon Cislo
 *
 * This file is part of FIRE++.
 *
 * FIRE++ is free software: you can redistribute it and/or modify it under the terms of the GNU
 * General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will by useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with this program.
 * If not, see <http://www.gnu.org/licenses/>
 *
 ***********************************************************************************************/

/*********************************************************************************************//**
 * 	\file MolecularDynamics.h
 * 	\brief Molecular dynamics integrators to find updated particle positions.
 *
 * 	I should write a detailed explanation at some point.
 *
 * 	\author Dillon Cislo
 * 	\date 11/16/2018
 * 	\copyright GNU Public License
 *
 ************************************************************************************************/

#ifndef _MOLECULAR_DYNAMICS_H_
#define _MOLECULAR_DYNAMICS_H_

#include <Eigen/Core>
#include <stdexcept> // std::runtime_error
#include <iostream>
#include <cmath>

namespace FIREpp {

///
/// Molecular dynamics integrators to find updated system vectors. Mainly for internal use.
///
template <typename Scalar>
class MolecularDynamics {

	private:

		typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> 		Vector;
		typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> 	Matrix;
		typedef Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic> 	Array;

	public:

		///
		/// Molecular dynamics via velocity verlet integration
		///
		/// \param f 		A function object such that `f(x, grad)` returns the
		/// 			objective function value at `x`, and overwrites `grad`
		///			with the gradient.
		/// \param fx 		IN: The objective function value at the current point.
		///			OUT: The function value at the new point.
		/// \param x 		IN: The current point.
		///			OUT: The new point.
		/// \param v 		IN: The current velocity.
		/// 			OUT: The new velocity.
		/// \param grad		IN: The gradient at the current point.
		/// 			OUT: The gradient at the new point.
		/// \param dt 		The current time step
		///
		/// \param alpha 	The current steering parameter
		///
		/// \param m 		The vector of particle masses
		///
		/// \param param	Parameters for the FIRE algorithm
		template <typename Foo>
		static void VelocityVerlet( Foo &f, Scalar &fx,
				Vector &x, Vector &v, Vector &grad,
				const Scalar &dt, const Scalar &alpha,
				const Vector &m,
				const FIREParam<Scalar> &param ) {

			// Check input arguments
			if ( dt <= Scalar(0) )
				std::invalid_argument("'dt' must be positive");
			if ( alpha <= Scalar(0) )
				std::invalid_argument("'alpha' must be positive");

			// Choose appropriate integration algorithm
			switch( param.boundary_conditions ) {

				case FIRE_NO_BOUNDARY_CONDITIONS : 
					
					VelocityVerlet_NBC( f, fx,
					x, v, grad, dt, alpha, m, param );
					break;

				case FIRE_PERIODIC_BOUNDARY_CONDITIONS :
					
					VelocityVerlet_PBC( f, fx,
					x, v, grad, dt, alpha, m, param );
					break;

				case FIRE_HARD_BOUNDARY_CONDITIONS :
					
					VelocityVerlet_HBC( f, fx,
					x, v, grad, dt, alpha, m, param );
					break;

			}

		};

		//! Velocity verlet integration with no boundary conditions
		template <typename Foo>
		static void VelocityVerlet_NBC( Foo &f, Scalar &fx,
				Vector &x, Vector &v, Vector &grad,
				const Scalar &dt, const Scalar &alpha,
				const Vector &m,
				const FIREParam<Scalar> &param ) {

			// Perform the first set of position/velocity updates 
			Vector a = -( grad.array() / m.array() ).matrix();
			x +=  dt * v + dt * dt * a / Scalar(2.0);
			v +=  dt * a / Scalar(2.0);

			// Update gradient vector
			fx = f( x, grad );

			// Perform final velocity updates
			a = -( grad.array() / m.array() ).matrix();
			v += dt * a / Scalar(2.0);
			
		};

		//! Velocity verlet integration with periodic boundary conditions
		template <typename Foo>
		static void VelocityVerlet_PBC( Foo &f, Scalar &fx,
				Vector &x, Vector &v, Vector &grad,
				const Scalar &dt, const Scalar &alpha,
				const Vector &m,
				const FIREParam<Scalar> &param ) {

			// Extract bounding box
			Vector lbnd = param.lbnd;
			Vector ubnd = param.ubnd;
			Vector len = ubnd - lbnd;

			// Perform the first set of position/velocity updates 
			Vector a = -( grad.array() / m.array() ).matrix();
			x +=  dt * v + dt * dt * a / Scalar(2.0);
			v +=  dt * a / Scalar(2.0);

			// Reset positions within the periodic bounding box
			for( int i = 0; i < param.n_p; i++ ) {
				for( int j = 0; j < param.dim; j++ ) {

					/*
					if ( x( i + j*param.n_p ) > ubnd(j) ) {
						x( i + j*param.n_p ) -= len(j);
					}

					if ( x( i + j*param.n_p ) < lbnd(j) ) {
						x( i + j*param.n_p ) += len(j);
					}
					*/

					x(i+j*param.n_p) -= len(j) * 
					std::floor( ( x(i+j*param.n_p)-lbnd(j) ) / len(j) );

				}

			}

			// Update gradient vector
			fx = f( x, grad );

			// Perform final velocity updates
			a = -( grad.array() / m.array() ).matrix();
			v += dt * a / Scalar(2.0);

		};

		//! Velocity verlet integration with hard wall boundary conditions
		template <typename Foo>
		static void VelocityVerlet_HBC( Foo &f, Scalar &fx,
				Vector &x, Vector &v, Vector &grad,
				const Scalar &dt, const Scalar &alpha,
				const Vector &m,
				const FIREParam<Scalar> &param ) {

			std::cerr << "This functionality is currently unsupported!" << std::endl;

		};

		

};

} // namespace FIREpp

#endif // _MOLECULAR_DYNAMICS_H_

/*

		//! Velocity verlet integration with no boundary conditions
		template <typename Foo>
		static void VelocityVerlet_NBC( Foo &f, Scalar &fx,
				Vector &x, Vector &v, Vector &grad,
				const Scalar &dt, const Scalar &alpha,
				const Vector &m,
				const FIREParam<Scalar> &param ) {

			Vector xi(3);
			Vector vi(3);
			Vector Fi(3);
		       	Vector ai(3);

			// ---------------------------------------------------------------------
			// Perform the first set of position/velocity updates 
			// ---------------------------------------------------------------------
			for( int i = 0; i < param.n_p; i++ ) {

				// Find x(t), v(t), F(t)
				for( int j = 0; j < param.dim; j++ ) {

					xi(j) = x(i+j*param.n_p);
					vi(j) = v(i+j*param.n_p);
					Fi(j) = Scalar(-1.0)*grad(i+j*param.n_p);

				}

				// Calculate a(t)
				ai = Fi / m(i);

				// Calculate x(t+dt)
				xi = xi + dt * vi + dt * dt * ai / Scalar(2.0);

				// Partially v(t)<-v*(t)
				vi = vi + dt * ai / Scalar(2.0);

				// Update position vector
				for( int j = 0; j < param.dim; j++ ) {

					x(i+j*param.n_p) = xi(j);
					v(i+j*param.n_p) = vi(j);
				}

			}

			// ---------------------------------------------------------------------
			// Update gradient vector
			// ---------------------------------------------------------------------
			fx = f( x, grad );

			// ---------------------------------------------------------------------
			// Perform final velocity updates
			// ---------------------------------------------------------------------
			for( int i = 0; i < param.n_p; i++ ) {

				// Find v*(t), F(t+dt)
				for( int j = 0; j < param.dim; j++ ) {

					vi(j) = v(i+j*param.n_p);
					Fi(j) = Scalar(-1.0)*grad(i+j*param.n_p);

				}

				// Calcualte a(t+dt)
				ai = Fi / m(i);

				// Calculate v(t+dt)
				vi = vi + dt * ai / Scalar(2.0);

				// Update velocity vector
				for( int j = 0; j < param.dim; j++ ) {

					v(i+j*param.n_p) = vi(j);

				}

			}
			
		};

*/
