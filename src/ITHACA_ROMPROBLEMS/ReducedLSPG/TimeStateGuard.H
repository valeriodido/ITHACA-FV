/*---------------------------------------------------------------------------*\
     ██╗████████╗██╗  ██╗ █████╗  ██████╗ █████╗       ███████╗██╗   ██╗
     ██║╚══██╔══╝██║  ██║██╔══██╗██╔════╝██╔══██╗      ██╔════╝██║   ██║
     ██║   ██║   ███████║███████║██║     ███████║█████╗█████╗  ██║   ██║
     ██║   ██║   ██╔══██║██╔══██║██║     ██╔══██║╚════╝██╔══╝  ╚██╗ ██╔╝
     ██║   ██║   ██║  ██║██║  ██║╚██████╗██║  ██║      ██║      ╚████╔╝
     ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝      ╚═╝       ╚═══╝

 * In real Time Highly Advanced Computational Applications for Finite Volumes
 * Copyright (C) 2026 by the ITHACA-FV authors
-------------------------------------------------------------------------------
License
    This file is part of ITHACA-FV
    ITHACA-FV is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    ITHACA-FV is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with ITHACA-FV. If not, see <http://www.gnu.org/licenses/>.
Class
    TimeStateGuard
Description
    Saves the time value, time index and time step of a Time object, and restores them when it goes out
    of scope. For diagnostics that move the shared time (e.g. along a FOM trajectory).
\*---------------------------------------------------------------------------*/

#ifndef TimeStateGuard_H
#define TimeStateGuard_H

#include "Time.H"

class TimeStateGuard
{
public:
    explicit TimeStateGuard(Foam::Time& runTime)
        : runTime_(runTime),
          value_(runTime.value()),
          index_(runTime.timeIndex()),
          deltaT_(runTime.deltaTValue())
    {}

    TimeStateGuard(const TimeStateGuard&) = delete;
    TimeStateGuard& operator=(const TimeStateGuard&) = delete;

    ~TimeStateGuard()
    {
        runTime_.setDeltaT(deltaT_, false);
        runTime_.setTime(value_, index_);
    }

private:
    Foam::Time& runTime_;
    const Foam::scalar value_;
    const Foam::label index_;
    const Foam::scalar deltaT_;
};

#endif
