import React from "react";
import { useEffect } from "react";
import { useState } from "react";
import { createContext } from "react";

export const UserContext = createContext();

export const UserProvider = (props) => {
    const [token, setToken] = useState(localStorage.getItem("token"));

    useEffect(() => {
        const fetchUser = async () => {
            const requestOptions = {
                method: "GET",
                headeer: {
                    "Content-Type": "application/json",
                    Authorization: "Bearer " + token,
                },
            };

            const response = await fetch("/unprotected", requestOptions);
            if (!response.ok){
                setToken(null);
            }
            localStorage.setItem("token", token);
        };
        fetchUser();
    }, [token] );

        return(
            <UserContext.Provider value={[token, setToken]}>
                {props.children}
            </UserContext.Provider>
        );
};