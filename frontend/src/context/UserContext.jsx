import React from "react";
import { useEffect } from "react";
import { useState } from "react";
import { createContext } from "react";

export const UserContext = createContext();

export const UserProvider = (props) => {
    const [token, setToken] = useState(localStorage.getItem("Token"));

    useEffect(() => {
        const fetchUser = async () => {
            const requestOptions = {
                method: "GET",
                headeer: {
                    "Content-Type": "application/json",
                    // Authorization: "Bearer " + token,
                },
            };

            const response = await fetch("/data", requestOptions);
            if (!response.ok){
                setToken(null);
            }
            localStorage.setItem("Token", token);
        };
        fetchUser();
    }, [token] );

        return(
            <UserContext.Provider value={[token, setToken]}>
                {props.children}
            </UserContext.Provider>
        );
};